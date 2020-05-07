# Importing the libraries
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
it = 0

class ReplayBuffer(object):
  def __init__(self, max_size=10000, sample_size = 500):
    self.storage = []
    self.max_size = max_size
    self.sample_size = sample_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.ptr = (self.ptr) % self.max_size
      self.storage[int(self.ptr)] = transition
    else:
      self.storage.append(transition)
    self.ptr+=1

  def sample(self, batch_size):
    if len(self.storage) < batch_size * 10:
        ind = np.random.randint(0, len(self.storage), size=batch_size)
    else:
        ind_range_low = np.random.randint(0, len(self.storage) - self.sample_size)
        ind_range_high = ind_range_low + self.sample_size

        ind = []
        while len(ind) < batch_size:
            ind_elm = np.random.randint(ind_range_low, ind_range_high)
            if ind_elm not in ind:
                ind.append(ind_elm)
        ind = np.array(ind, copy=False)

    batch_states_img, batch_states_ori1, batch_states_ori2, batch_states_dis,\
    batch_next_states_img, batch_next_states_ori1, batch_next_states_ori2, batch_next_states_dis,\
    batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], [], [], [], [], []

    for i in ind:
      state_img,state_ori1,state_ori2,state_dis,\
      next_state_img,next_state_ori1,next_state_ori2,next_state_dis,\
      action, reward, done = self.storage[i]

      batch_states_img.append(np.array(state_img, copy=False))
      batch_states_ori1.append(np.array(state_ori1, copy=False))
      batch_states_ori2.append(np.array(state_ori2, copy=False))
      batch_states_dis.append(np.array(state_dis, copy=False))

      batch_next_states_img.append(np.array(next_state_img, copy=False))
      batch_next_states_ori1.append(np.array(next_state_ori1, copy=False))
      batch_next_states_ori2.append(np.array(next_state_ori2, copy=False))
      batch_next_states_dis.append(np.array(next_state_dis, copy=False))


      batch_actions.append(action)
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))

    return batch_states_img, batch_states_ori1, batch_states_ori2, batch_states_dis,\
    batch_next_states_img, batch_next_states_ori1, batch_next_states_ori2, batch_next_states_dis,\
    batch_actions, np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# 3*3 convolutino
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class Actor(nn.Module):
    def __init__(self, num_classes=1, max_action=5):
        block, layers = ResidualBlock, [2,2,2,2]
        super(Actor, self).__init__()
        self.in_channels = 8

        self.conv = conv3x3(1, self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, self.in_channels, layers[0], 2)
        self.layer3 = self.make_layer(block, self.in_channels, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(12)
        self.fc1 = nn.Linear(self.in_channels, 20)
        self.fc2 = nn.Linear(23, 10)
        self.fc3 = nn.Linear(10, num_classes)
    
        self.max_action = max_action

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, ori1, ori2, dis):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.cat([out, ori1, ori2, dis], 1)
        out = self.fc2(out)
        out = self.max_action * torch.tanh(self.fc3(out))

        rotation = out.clone().detach().reshape(out.shape[0], -1)
        return rotation

class Critic(nn.Module):
    def __init__(self):
        block, layers = ResidualBlock, [2,2,2,2]
        super(Critic, self).__init__()
        self.in_channels = 8

        self.conv1 = conv3x3(1, self.in_channels)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, self.in_channels, layers[0], 2)
        self.layer3 = self.make_layer(block, self.in_channels, layers[1], 2)
        self.avg_pool1 = nn.AvgPool2d(12)
        self.fc1 = nn.Linear(self.in_channels, 20)
        self.fc2 = nn.Linear(24, 10)
        self.fc3 = nn.Linear(10, 1)

        self.conv2 = conv3x3(1, self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer4 = self.make_layer(block, self.in_channels, layers[0])
        self.layer5 = self.make_layer(block, self.in_channels, layers[0], 2)
        self.layer6 = self.make_layer(block, self.in_channels, layers[1], 2)
        self.avg_pool2 = nn.AvgPool2d(12)
        self.fc4 = nn.Linear(self.in_channels, 20)
        self.fc5 = nn.Linear(24, 10)
        self.fc6 = nn.Linear(10, 1)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, ori1, ori2, dis, action):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.avg_pool1(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = torch.cat([out1, ori1, ori2, dis, action], 1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = self.layer4(out2)
        out2 = self.layer5(out2)
        out2 = self.layer6(out2)
        out2 = self.avg_pool2(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc4(out2)
        out2 = torch.cat([out2, ori1, ori2, dis, action], 1)
        out2 = self.fc5(out2)
        out2 = self.fc6(out2)

        return out1, out2

    def Q1(self, x, ori1, ori2, dis, action):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.avg_pool1(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = torch.cat([out1, ori1, ori2, dis, action], 1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)

        return out1

# class Actor(nn.Module):
#   def __init__(self, state_dim=5, action_dim=5, max_action=5):
#     super(Actor, self).__init__()
#     #MP
#     self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
#     self.norm1 = nn.BatchNorm2d(8)
#     self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm2 = nn.BatchNorm2d(16)
#     self.conv3 = nn.Conv2d(16, 8, kernel_size=1)
#     self.norm3 = nn.BatchNorm2d(8)
#     self.drop1 = nn.Dropout2d()
#     # MP
#     self.conv4 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm4 = nn.BatchNorm2d(16)
#     self.conv5 = nn.Conv2d(16, 16, kernel_size=3)
#     self.norm5 = nn.BatchNorm2d(16)
#     self.conv6 = nn.Conv2d(16, 8, kernel_size=1)
#     self.norm6 = nn.BatchNorm2d(8)
#     self.drop2 = nn.Dropout2d()

#     self.conv7 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm7 = nn.BatchNorm2d(16)
#     self.conv8 = nn.Conv2d(16, 16, kernel_size=3)
#     self.norm8 = nn.BatchNorm2d(16)

#     self.fc1 = nn.Linear(16, 10)
#     self.fc2 = nn.Linear(13, 20)
#     self.fc3 = nn.Linear(20, action_dim)
#     self.max_action = max_action

#   def forward(self, state_img, state_ori1, state_ori2, state_dis):
#     x = F.relu(self.norm1(self.conv1(state_img))) #48
#     x = F.relu(self.norm2(self.conv2(x))) #46

#     x = F.relu(self.norm3(self.conv3(x))) #46
#     x = F.relu(F.max_pool2d(self.drop1(x), 2)) #23

#     x = F.relu(self.norm4(self.conv4(x))) #21
#     x = F.relu(self.norm5(self.conv5(x))) #19

#     x = F.relu(self.norm6(self.conv6(x))) #19
#     x = F.relu(F.max_pool2d(self.drop2(x), 2)) #9

#     x = F.relu(self.norm7(self.conv7(x))) #7
#     x = F.relu(self.norm8(self.conv8(x))) #5

#     x = F.avg_pool2d(x, (5, 5))
#     x = x.reshape(x.size(0), -1) # 441
#     x = F.relu(self.fc1(x)) # 10
#     x = torch.cat([x, state_ori1, state_ori2, state_dis], 1) #13
#     x = F.relu(self.fc2(x)) # 20
#     x = F.dropout(x, training=self.training)
#     x = self.max_action * torch.tanh(x)#/ (1+torch.exp(self.fc3(x))) #3

#     x_copy = x.clone().detach().reshape(x.shape[0], -1)
#     rotation = x_copy
#     return rotation

# # Step 3: We build two neural networks for the two Critic models
# # and two neural networks for the two Critic targets
# class Critic(nn.Module):

#   def __init__(self, state_dim=5, action_dim=9):
#     super(Critic, self).__init__()

#     #MP
#     self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
#     self.norm1 = nn.BatchNorm2d(8)
#     self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm2 = nn.BatchNorm2d(16)
#     self.conv3 = nn.Conv2d(16, 8, kernel_size=1)
#     self.norm3 = nn.BatchNorm2d(8)
#     self.drop1 = nn.Dropout2d()
    
#     self.conv4 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm4 = nn.BatchNorm2d(16)
#     self.conv5 = nn.Conv2d(16, 16, kernel_size=3)
#     self.norm5 = nn.BatchNorm2d(16)
#     self.conv6 = nn.Conv2d(16, 8, kernel_size=1)
#     self.norm6 = nn.BatchNorm2d(8)
#     self.drop2 = nn.Dropout2d()

#     self.conv7 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm7 = nn.BatchNorm2d(16)
#     self.conv8 = nn.Conv2d(16, 16, kernel_size=3)
#     self.norm8 = nn.BatchNorm2d(16)

#     self.fc1 = nn.Linear(16, 10)
#     #concat
#     self.fc2 = nn.Linear(14, 20)
#     self.fc3 = nn.Linear(20, 1)

#     ###################

#     #MP
#     self.conv9 = nn.Conv2d(1, 8, kernel_size=3)
#     self.norm9 = nn.BatchNorm2d(8)
#     self.conv10 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm10 = nn.BatchNorm2d(16)
#     self.conv11 = nn.Conv2d(16, 8, kernel_size=1)
#     self.norm11 = nn.BatchNorm2d(8)
#     self.drop3 = nn.Dropout2d()

#     self.conv12 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm12 = nn.BatchNorm2d(16)
#     self.conv13 = nn.Conv2d(16, 16, kernel_size=3)
#     self.norm13 = nn.BatchNorm2d(16)
#     self.conv14 = nn.Conv2d(16, 8, kernel_size=1)
#     self.norm14 = nn.BatchNorm2d(8)
#     self.drop4 = nn.Dropout2d()

#     self.conv15 = nn.Conv2d(8, 16, kernel_size=3)
#     self.norm15 = nn.BatchNorm2d(16)
#     self.conv16 = nn.Conv2d(16, 16, kernel_size=3)
#     self.norm16 = nn.BatchNorm2d(16)

#     self.fc4 = nn.Linear(16, 10)
#     #concat
#     self.fc5 = nn.Linear(14, 20)
#     self.fc6 = nn.Linear(20, 1)


#   def forward(self, state_img, state_ori1, state_ori2, state_dis, action):
#     x1 = F.relu(self.norm1(self.conv1(state_img))) #48
#     x1 = F.relu(self.norm2(self.conv2(x1))) #46

#     x1 = F.relu(self.norm3(self.conv3(x1))) #46
#     x1 = F.relu(F.max_pool2d(self.drop1(x1), 2)) #23

#     x1 = F.relu(self.norm4(self.conv4(x1))) #21
#     x1 = F.relu(self.norm5(self.conv5(x1))) #19

#     x1 = F.relu(self.norm6(self.conv6(x1))) #19
#     x1 = F.relu(F.max_pool2d(self.drop2(x1), 2)) #9

#     x1 = F.relu(self.norm7(self.conv7(x1))) #7
#     x1 = F.relu(self.norm8(self.conv8(x1))) #5

#     x1 = F.avg_pool2d(x1, (5, 5))
#     x1 = x1.reshape(x1.size(0), -1) # 441
#     x1 = F.relu(self.fc1(x1)) # 10
#     x1 = torch.cat([x1, state_ori1, state_ori2, state_dis, action], 1) #13
#     x1 = F.relu(self.fc2(x1)) # 20
#     x1 = F.dropout(x1, training=self.training)
#     x1 = self.fc3(x1) # 1

#     #################
#     x2 = F.relu(self.norm9(self.conv9(state_img))) #48
#     x2 = F.relu(self.norm10(self.conv10(x2))) #46

#     x2 = F.relu(self.norm11(self.conv11(x2))) #46
#     x2 = F.relu(F.max_pool2d(self.drop3(x2), 2)) #23

#     x2 = F.relu(self.norm12(self.conv12(x2))) #21
#     x2 = F.relu(self.norm13(self.conv13(x2))) #19

#     x2 = F.relu(self.norm14(self.conv14(x2))) #19
#     x2 = F.relu(F.max_pool2d(self.drop4(x2), 2)) #9

#     x2 = F.relu(self.norm15(self.conv15(x2))) #7
#     x2 = F.relu(self.norm16(self.conv16(x2))) #5

#     x2 = F.avg_pool2d(x2, (5, 5))
#     x2 = x2.reshape(x2.size(0), -1) #5
#     x2 = F.relu(self.fc4(x2)) # 10
#     x2 = torch.cat([x2, state_ori1, state_ori2, state_dis, action], 1) #13
#     x2 = F.relu(self.fc5(x2)) # 20
#     x2 = F.dropout(x2, training=self.training)
#     x2 = self.fc6(x2) # 1

#     return x1, x2

  # def Q1(self, state_img, state_ori1, state_ori2, state_dis, action):
  #   x1 = F.relu(self.norm1(F.max_pool2d(self.conv1(state_img), 2))) #24
  #   x1 = F.relu(self.norm2(self.conv2(x1))) #22
  #   x1 = F.relu(F.max_pool2d(self.drop1(x1), 2)) #11

  #   x1 = F.relu(self.norm3(self.conv3(x1))) #11
  #   x1 = F.relu(self.norm4(self.conv4(x1))) #9
  #   x1 = self.drop2(x1)
  #   x1 = F.relu(self.norm5(self.conv5(x1))) #7

  #   x1 = F.avg_pool2d(x1, (7, 7))
  #   x1 = x1.reshape(x1.size(0), -1) #5
  #   x1 = F.relu(self.fc1(x1)) # 10
  #   x1 = torch.cat([x1, state_ori1, state_ori2, state_dis, action], 1) # 14
  #   x1 = F.relu(self.fc2(x1)) #20
  #   x1 = F.dropout(x1, training=self.training)
  #   x1 = self.fc3(x1) # 1
  #   return x1

# def freeze_layer(layer):
#     for param in layer.parameters():
#         param.requires_grad = False

# Building the whole Training Process into a class
# class TD3(object):

    # def __init__(self, state_dim, action_dim, max_action):
    #     self.actor = Actor(state_dim, action_dim, max_action)
    #     self.actor_target = Actor(state_dim, action_dim, max_action)
    #     self.actor_target.load_state_dict(self.actor.state_dict())
    #     self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)
    #     self.critic = Critic(state_dim, action_dim)
    #     self.critic_target = Critic(state_dim, action_dim)
    #     self.critic_target.load_state_dict(self.critic.state_dict())
    #     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01)
    #     self.max_action = max_action

        # my_layers = actor.layers()
        # for i in range(5):
        #     freeze_layer(my_layers[i])
        # my_layers = critic.layers()
        # for i in range(5):
        #     freeze_layer(my_layers[i])

class TD3(object):

    def __init__(self, action_dim, max_action):
        self.actor = Actor(action_dim, max_action)
        self.actor_target = Actor(action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01, weight_decay=1e-2)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.01, weight_decay=1e-2)

        self.max_action = max_action

    def select_action(self, state_img,state_ori1,state_ori2,state_dis):
        state_img = torch.Tensor(np.expand_dims(state_img, axis=0))
        state_img = torch.Tensor(np.expand_dims(state_img, axis=0))
        state_ori1 = torch.Tensor(np.expand_dims(state_ori1, axis=0))
        state_ori1 = torch.Tensor(np.expand_dims(state_ori1, axis=0))
        state_ori2 = torch.Tensor(np.expand_dims(state_ori2, axis=0))
        state_ori2 = torch.Tensor(np.expand_dims(state_ori2, axis=0))
        state_dis = torch.Tensor(np.expand_dims(state_dis, axis=0))
        state_dis = torch.Tensor(np.expand_dims(state_dis, axis=0))
        return self.actor(state_img,state_ori1,state_ori2,state_dis).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        global it
        it += 1

        # Step 4: We sample a batch of transitions (s, s', a, r) from the memory
        batch_states_img,batch_states_ori1,batch_states_ori2,batch_states_dis,\
        batch_next_states_img,batch_next_states_ori1,batch_next_states_ori2,batch_next_states_dis,\
        batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)

        state_img = torch.Tensor(np.expand_dims(batch_states_img, axis=1))
        state_ori1 = torch.Tensor(np.expand_dims(batch_states_ori1, axis=1))
        state_ori2 = torch.Tensor(np.expand_dims(batch_states_ori2, axis=1))
        state_dis = torch.Tensor(np.expand_dims(batch_states_dis, axis=1))

        next_state_img = torch.Tensor(np.expand_dims(batch_next_states_img, axis=1))
        next_state_ori1 = torch.Tensor(np.expand_dims(batch_next_states_ori1, axis=1))
        next_state_ori2 = torch.Tensor(np.expand_dims(batch_next_states_ori2, axis=1))
        next_state_dis = torch.Tensor(np.expand_dims(batch_next_states_dis, axis=1))

        action = torch.Tensor(np.expand_dims(batch_actions, axis=1))
        reward = torch.Tensor(batch_rewards)
        done = torch.Tensor(list(batch_dones))

        # Step 5: From the next state s', the Actor target plays the next action a'
        next_action = self.actor_target(next_state_img, next_state_ori1, next_state_ori2, next_state_dis)

        # # Step 6: We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment
        # noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise)
        # noise = noise.clamp(-noise_clip, noise_clip)
        # noise = noise.unsqueeze(1)

        # next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

        # Step 7: The two Critic targets take each the couple (s', a') as input and return two Q-values Qt1(s',a') and Qt2(s',a') as outputs
        target_Q1, target_Q2 = self.critic_target(next_state_img, next_state_ori1, next_state_ori2, next_state_dis, next_action)

        # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
        target_Q = torch.min(target_Q1, target_Q2)

        # Step 9: We get the final target of the two Critic models
        target_Q = reward + ((1 - done) * discount * target_Q).detach()

        # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
        current_Q1, current_Q2 = self.critic(state_img,state_ori1,state_ori2,state_dis, action)

        # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
        if it % policy_freq == 0:
          actor_loss = -self.critic.Q1(state_img,state_ori1,state_ori2,state_dis, self.actor(state_img,state_ori1,state_ori2,state_dis)).mean()
          self.actor_optimizer.zero_grad()
          actor_loss.backward()
          self.actor_optimizer.step()

          # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
          for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

          # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
          for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 -  tau) * target_param.data)

  # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

  # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))