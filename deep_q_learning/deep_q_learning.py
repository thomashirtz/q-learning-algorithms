import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
import numpy as np
from itertools import count
from more_itertools import pairwise
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
NETWORK_UPDATE_FREQUENCY = 1000

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


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


class DeepQNetwork(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=None):
        super().__init__()

        hidden_units = hidden_units or [128, 128]
        units = [input_features] + hidden_units + [output_features]
        self.layers = nn.ModuleList()
        for i, o in pairwise(units):
            self.layers.append(nn.Linear(i, o))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class EpsilonGreedyPolicy:
    def __init__(self, local_network, action_space, epsilon_start, epsilon_final, epsilon_decay):
        self.step = 0
        self.local_network = local_network
        self.actions = range(action_space)
        self.epsilon_start, self.epsilon_final, self.epsilon_decay = epsilon_start, epsilon_final, epsilon_decay

    def choose_action(self, observation, epsilon=None):
        epsilon = epsilon or self.get_epsilon()
        self.step += 1
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.local_network(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
                return q_values.max(1)[1].item()
        else:
            return random.choice(self.actions)

    def get_epsilon(self):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * self.step / self.epsilon_decay)


class DeepQLearning:
    def __init__(self, policy: EpsilonGreedyPolicy, target_network: nn.Module, optimizer, memory_capacity, batch_size):
        self.step = 0
        self.policy = policy
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.target_network = target_network
        self.memory = Memory(memory_capacity)

    def train(self, total_steps):
        env = gym.make(ENVIRONMENT)
        self.update_target_network()
        recent_scores = deque(maxlen=100)
        recent_losses = deque(maxlen=100)

        for episode in count():
            state, score = env.reset(), 0

            for _ in count():
                action = self.policy.choose_action(state)
                next_state, reward, done, _ = env.step(action)

                if type(next_state) is not np.ndarray:
                    next_state = np.array(state)
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

            if self.step > total_steps:
                break

            recent_scores.append(score)
            print(f'Step:{self.step}\tEpisode:{episode}   Score:{np.mean(recent_scores):.2f}   Loss{np.mean(recent_losses or [0]):.5f}   Epsilon{self.policy.get_epsilon():.2f}')

    def learn(self):
        batch = self.memory.sample(self.batch_size)

        done = torch.FloatTensor(batch.done)
        action = torch.LongTensor(batch.action)
        reward = torch.FloatTensor(batch.reward)
        state = torch.FloatTensor(np.float32(batch.state))
        next_state = torch.FloatTensor(np.float32(batch.next_state))

        q_values = self.policy.local_network(state)
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
        self.target_network.load_state_dict(self.policy.local_network.state_dict())


if __name__ == '__main__':
    local_network = DeepQNetwork(OBSERVATION_SPACE, ACTION_SPACE)
    target_network = DeepQNetwork(OBSERVATION_SPACE, ACTION_SPACE)
    target_network.eval()
    local_network_optimizer = optim.Adam(local_network.parameters(), LEARNING_RATE)

    policy = EpsilonGreedyPolicy(local_network, ACTION_SPACE, EPSILON_START, EPSILON_FINAL, EPSILON_DECAY)
    trainer = DeepQLearning(policy, target_network, local_network_optimizer, MEMORY_CAPACITY, BATCH_SIZE)
    trainer.train(40000)
