import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
from itertools import count
from collections import deque, namedtuple

from per import ProportionalPrioritizedMemory

import matplotlib.pyplot as plt

# Environment Parameters
ENVIRONMENT = 'CartPole-v0'
environment = gym.make(ENVIRONMENT)
ACTION_SPACE = environment.action_space.n
OBSERVATION_SPACE = environment.observation_space.shape[0]
del environment

# Exploration-related Parameters
EPSILON_START = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY = 1000

# Memory Related Parameters
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

# QLearning Parameters
GAMMA = 0.99

# Prioritized Replay Paramteres
ALPHA = 0.6
BETA = 0.4

# Network Parameters
HIDDEN_UNITS = 128
LEARNING_RATE = 0.0001
NETWORK_UPDATE_FREQUENCY = 1000

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


def get_epsilon(step):
    return EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * np.exp(-1. * step / EPSILON_DECAY)


class DQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc_1 = nn.Linear(input_features, HIDDEN_UNITS)
        self.fc_2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc_3 = nn.Linear(HIDDEN_UNITS, output_features)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return self.fc_3(x)


class Agent:
    def __init__(self, local_network, action_space):
        self.local_network = local_network
        self.actions = range(action_space)

    def __call__(self, observation, epsilon=0.):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.local_network(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
                return q_values.max(1)[1].item()
        else:
            return random.choice(self.actions)


def get_networks_and_optimizer(observation_space, action_space, learning_rate):
    online_network = DQN(observation_space, action_space)
    target_network = DQN(observation_space, action_space)
    target_network.eval()
    online_network_optimizer = optim.Adam(online_network.parameters(), learning_rate)
    return online_network, target_network, online_network_optimizer


class DeepQLearning:
    def __init__(self, agent: Agent, target_network: nn.Module, optimizer, memory_capacity, batch_size):
        self.step = 0
        self.agent = agent
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.target_network = target_network
        self.memory = ProportionalPrioritizedMemory(memory_capacity)

    def train(self, total_step):
        env = gym.make(ENVIRONMENT)
        self.update_target_network()
        recent_scores = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        plot = []

        for episode in count():
            state = env.reset()
            score, i, epsilon = 0, 0, 0
            for i in count():
                epsilon = get_epsilon(self.step)
                action = self.agent(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                self.memory.push(Experience(state, action, reward, next_state, done))

                self.step += 1
                score += reward
                state = next_state

                if len(self.memory) > self.batch_size:
                    loss = self.learn()
                    recent_losses.append(loss)

                if self.step % NETWORK_UPDATE_FREQUENCY == 0:
                    self.update_target_network()

                if self.step % 100 == 0:
                    plot.append(np.mean(recent_scores))

                if done:
                    break

            recent_lengths.append(i)
            recent_scores.append(score)
            print(self.step, episode, np.mean(recent_scores), np.mean(recent_lengths), np.mean(recent_losses), epsilon)

            if self.step > total_step:
                break

        plt.plot(plot)
        plt.show()


    def learn(self):
        indexes, weights, experiences = self.memory.sample(self.batch_size)

        batch = Experience(*zip(*experiences))
        weights = torch.tensor(weights)

        done = torch.FloatTensor(batch.done)
        action = torch.LongTensor(batch.action)
        reward = torch.FloatTensor(batch.reward)
        state = torch.FloatTensor(np.float32(batch.state))
        next_state = torch.FloatTensor(np.float32(batch.next_state))

        q_values = self.agent.local_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + GAMMA * next_q_value * (1 - done)

        delta = q_value - expected_q_value.detach()
        self.memory.update(deltas=delta.tolist(), indexes=indexes)

        loss = (delta.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.agent.local_network.state_dict())


if __name__ == '__main__':
    for i in range(5):
        local_network, target_network, local_network_optimizer = \
            get_networks_and_optimizer(OBSERVATION_SPACE, ACTION_SPACE, LEARNING_RATE)
        agent = Agent(local_network, ACTION_SPACE)
        trainer = DeepQLearning(agent, target_network, local_network_optimizer, MEMORY_CAPACITY, BATCH_SIZE)
        trainer.train(50000)

    env = gym.make(ENVIRONMENT)
    env = gym.wrappers.Monitor(env, "./video", force=True)
    state = env.reset()
    done = False
    while not done:
        action = agent(state, EPSILON_FINAL)
        next_state, reward, done, _ = env.step(action)
        state = next_state

