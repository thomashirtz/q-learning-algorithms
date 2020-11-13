import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

import gym
import sys
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
EPSILON_DECAY = 2000

TOTAL_STEP = 200000
NUM_PROCESSES = 4

GAMMA = 0.9
LEARNING_RATE = 0.0005

HIDDEN_UNITS = [128, 128]
OPTIMIZER_STEP_FREQUENCY = 10
UPDATE_TARGET_NETWORK_FREQUENCY = 100

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingDQN(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=None):
        super().__init__()

        hidden_units = hidden_units or [128, 128]
        units = [input_features] + hidden_units
        self.layers = nn.ModuleList()
        for i, o in pairwise(units):
            self.layers.append(nn.Linear(i, o))

        self.fc_value = nn.Linear(hidden_units[-1], 1)
        self.fc_advantage = nn.Linear(hidden_units[-1], output_features)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class EpsilonGreedyPolicy:
    def __init__(self, online_network, action_space, epsilon_start, epsilon_final, epsilon_decay):
        self.step = 0
        self.online_network = online_network
        self.actions = range(action_space)
        self.epsilon_start, self.epsilon_final, self.epsilon_decay = epsilon_start, epsilon_final, epsilon_decay

    def choose_action(self, observation, epsilon=None):
        epsilon = epsilon or self.get_epsilon(self.step)
        self.step += 1
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.online_network(torch.tensor(observation, dtype=torch.float32).unsqueeze(0))
                return q_values.max(1)[1].item()
        else:
            return random.choice(self.actions)

    def get_epsilon(self, step):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * step / self.epsilon_decay)


class Worker:
    def __init__(self, online_network, target_network, optimizer, rank):
        self.rank = rank
        self.optimizer = optimizer
        self.local_step = 0
        self.online_network = online_network
        self.target_network = target_network
        self.policy = EpsilonGreedyPolicy(online_network, ACTION_SPACE, EPSILON_START, EPSILON_FINAL, EPSILON_DECAY)

    def run(self, global_step):
        env = gym.make(ENVIRONMENT)
        recent_scores = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_losses = deque(maxlen=100)
        for episode in count():
            state = env.reset()
            score, step, epsilon = 0, 0, 0
            for step in count():

                with global_step.get_lock():
                    global_step.value += 1
                self.local_step += 1

                action = self.policy.choose_action(state, epsilon)
                next_state, reward, done, _ = env.step(action)

                if type(next_state) is not np.ndarray:
                    next_state = np.array(state)
                experience = Experience(state, action, reward, next_state, done)

                self.accumulate_gradient(experience)

                score += reward
                state = next_state

                if step % OPTIMIZER_STEP_FREQUENCY == 0:
                    self.optimizer_step()

                if step % UPDATE_TARGET_NETWORK_FREQUENCY == 0:
                    self.update_target_network()

                if done:
                    break

            recent_lengths.append(step)
            recent_scores.append(score)
            print(self.local_step, episode, np.mean(recent_scores), np.mean(recent_lengths))
            sys.stdout.flush() # https://stackoverflow.com/questions/2774585/child-processes-created-with-python-multiprocessing-module-wont-print

            if global_step.value > TOTAL_STEP:
                break
    def accumulate_gradient(self, experience):
        done = torch.FloatTensor([experience.done])
        action = torch.LongTensor([experience.action])
        reward = torch.FloatTensor([experience.reward])
        state = torch.FloatTensor(np.float32([experience.state]))
        next_state = torch.FloatTensor(np.float32([experience.next_state]))

        q_values = self.online_network(state)
        next_q_values = self.target_network(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + GAMMA * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss.backward()

    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())


if __name__ == '__main__':
    online_network = DuelingDQN(OBSERVATION_SPACE, ACTION_SPACE, HIDDEN_UNITS)
    target_network = DuelingDQN(OBSERVATION_SPACE, ACTION_SPACE, HIDDEN_UNITS)
    target_network.load_state_dict(online_network.state_dict())
    online_network.share_memory()
    target_network.share_memory()
    target_network.eval()

    global_step = mp.Value('i', 0)
    optimizer = optim.SGD(online_network.parameters(), lr=LEARNING_RATE)

    num_processes = 4

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=Worker(online_network, target_network, optimizer, rank).run,
                       args=(global_step, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()