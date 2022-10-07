import numpy as np
import gym
from utils import *
from agent import *
from config import *

def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    # 定义rewards_log and average_log列表，并且初始化eps的值
    rewards_log = []
    average_log = []
    eps = eps_init
    # 开始num_episode的循环
    for i in range(1, 1 + num_episode):
        # 初始化episode_reward,done,state,t的值

        episodic_reward = 0
        done = False
        state = env.reset()
        t = 0

        # 开始每一个episode的steps即训练次数
        while not done and t < max_t:
            # 循环开始，
            # agent选择action
            # env通过action反馈next_state,reward,done,info,但是info不需要
            # 向agent中的Replay Buffer中的deque中存入state，action，next_state，done
            t += 1
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            # 一个个Episode组成了个体的经历（Experience）。我看到过的一些模型使用一个叫“Memory”的概念来记录个体既往的经历，
            # 其建模思想是Memory仅无序存储一系列的Transition，不使用Episode这一概念，不反映Transition对象之间的关联，
            # 这也是可以完成基于记忆的离线学习的强化学习算法的，甚至其随机采样过程更简单。
            # 不过我还是额外设计了Episode以及在此基础上的Experience。
            agent.memory.append((state, action, reward, next_state, done))

            # 每4次并且deque中存入的数量比agent的mini batch大于等于的时候agent开始学习
            # 并且利用tau更新Q_target网络的参数
            if t % 4 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                agent.soft_update(agent.tau)

            # 将下一个状态的值赋给state
            # 累计奖励
            state = next_state.copy()
            episodic_reward += reward

        # 向rewards_log列表中追加每一个episode的奖励
        rewards_log.append(episodic_reward)
        # 计算average的奖励，并追加写入列表average_log中
        average_log.append(np.mean(rewards_log[-100:]))
        # 打印第i个episode及其累积奖励和平均的奖励
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 50 == 0:
            print()

        # 更新epsilon的值，eps = max(eps * eps_decay, eps_min)
        eps = max(eps * eps_decay, eps_min)

    # 返回奖励rewards_log的list
    return rewards_log


if __name__ == '__main__':
    # 选择环境
    env = gym.make(RAM_ENV_NAME)
    # 设置agent
    agent = Agent(env.observation_space.shape[0], env.action_space.n, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE, False)
    # train并且得到rewards_log
    rewards_log = train(env, agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    # 保存.npy文件并且写入rewards_log
    np.save('{}_rewards.npy'.format(RAM_ENV_NAME), rewards_log)
    # 加载agent.Q_local到CPU
    agent.Q_local.to('cpu')
    # 保存训练好的权重文件
    torch.save(agent.Q_local.state_dict(), '{}_weights.pth'.format(RAM_ENV_NAME))