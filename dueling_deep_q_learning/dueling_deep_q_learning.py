import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
from itertools import count
from collections import deque, namedtuple

# Environment Parameters
ENVIRONMENT = 'CartPole-v0'
environment = gym.make(ENVIRONMENT)
ACTION_SPACE = environment.action_space.n
OBSERVATION_SPACE = environment.observation_space.shape[0]
del environment

# Exploration-related Parameters
EPSILON_START = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY = 500

# Memory Related Parameters
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32

# QLearning Parameters
GAMMA = 0.99

# Network Parameters
HIDDEN_UNITS = 128
LEARNING_RATE = 0.0001
NETWORK_UPDATE_FREQUENCY = 100

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


def get_epsilon(step):
    return EPSILON_FINAL + (EPSILON_START - EPSILON_FINAL) * np.exp(-1. * step / EPSILON_DECAY)


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc = nn.Linear(input_features, HIDDEN_UNITS)

        self.fc_advantage_1 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc_advantage_2 = nn.Linear(HIDDEN_UNITS, output_features)

        self.fc_value_1 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc_value_2 = nn.Linear(HIDDEN_UNITS, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        value = self.fc_value_2(F.relu(self.fc_value_1(x)))
        advantage = self.fc_advantage_2(F.relu(self.fc_advantage_1(x)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class Agent:
    def __init__(self, local_network, action_space):
        self.local_network = local_network
        self.actions = range(action_space)

    def __call__(self, observation, epsilon=0):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.local_network(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
                return q_values.max(1)[1].item()
        else:
            return random.choice(self.actions)


def get_networks_and_optimizer(observation_space, action_space, learning_rate):
    online_network = DuelingDQN(observation_space, action_space)
    target_network = DuelingDQN(observation_space, action_space)
    target_network.eval()
    online_network_optimizer = optim.Adam(online_network.parameters(), learning_rate)
    return online_network, target_network, online_network_optimizer


class QLearning:
    def __init__(self, agent: Agent, target_network: nn.Module, optimizer, memory_capacity, batch_size):
        self.step = 0
        self.agent = agent
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.target_network = target_network
        self.memory = Memory(memory_capacity)  # TODO add gamma

    def train(self, episodes):
        env = gym.make(ENVIRONMENT)
        self.update_target_network()
        recent_scores = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_losses = deque(maxlen=100)

        for episode in range(episodes):
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

                if len(self.memory.buffer) > self.batch_size:
                    loss = self.learn()
                    recent_losses.append(loss)

                if self.step % NETWORK_UPDATE_FREQUENCY == 0:
                    self.update_target_network()

                if done:
                    break

            recent_lengths.append(i)
            recent_scores.append(score)
            print(self.step, np.mean(recent_scores), np.mean(recent_lengths), np.mean(recent_losses), epsilon)

    def learn(self):
        batch = self.memory.sample(self.batch_size)

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

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.agent.local_network.state_dict())


if __name__ == '__main__':
    local_network, target_network, local_network_optimizer = \
        get_networks_and_optimizer(OBSERVATION_SPACE, ACTION_SPACE, LEARNING_RATE)
    agent = Agent(local_network, ACTION_SPACE)
    trainer = QLearning(agent, target_network, local_network_optimizer, MEMORY_CAPACITY, BATCH_SIZE)
    trainer.train(300)
