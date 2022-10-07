import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np

from networks import *

class Agent:

    def __init__(self, state_size, action_size, bs, lr, tau, gamma, device, visual=False):
        '''
        When dealing with visual inputs, state_size should work as num_of_frame
        初始化，状态size，动作size，batch_size，learning-rate，tau，gamma奖励衰减值，是否用GPU
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        # 选择使用visual-Q-Network还是Q-Network
        if visual:
            # Q_local用于预测Q估计,使用最新的参数；Q_target用于预测Q现实，使用很久之前的参数
            self.Q_local = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
            self.Q_target = Visual_Q_Network(self.state_size, self.action_size).to(self.device)
        else:
            self.Q_local = Q_Network(self.state_size, self.action_size).to(device)
            self.Q_target = Q_Network(self.state_size, self.action_size).to(device)
        # 使用soft_update更新Q_target网络的参数
        self.soft_update(1)
        # 使用Adam优化器优化预测Q_local
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        # 设置Replay Buffer
        self.memory = deque(maxlen=100000)

    # agent选择动作
    def act(self, state, eps=0):
        # 根据随机产生的数和epsilon比较，若是产生的大于epsilon，则选择action value最大的值的action
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        # 若是小于等于epsilon则随机选择一个动作
        else:
            return random.choice(np.arange(self.action_size))
    # agent学习
    def learn(self):
        # 从Replay Buffer中抽取mini-batch
        experiences = random.sample(self.memory, self.bs)
        # 读取deque中的experience中的索引为0,1,2,3,4的列加载到GPU计算
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)
        # 获取当前状态下的Q_local的值
        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)
        # 获取下一个状态下的目标网络Q_targets的值
        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            # 计算Q_target的值
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        # 计算loss=(Q_values - Q_targets).pow(2).mean()
        loss = (Q_values - Q_targets).pow(2).mean()
        # 反向传播更新梯度
        self.optimizer.zero_grad()
        loss.backward()
        # 每一步更新一次
        self.optimizer.step()

    # 初始化，让两个网络当前Q_local和目标优化网络Q_target权重一致
    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)