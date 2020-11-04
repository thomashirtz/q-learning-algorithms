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

# Memory Related Parameters
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

# QLearning Parameters
GAMMA = 0.99

# Network Parameters
HIDDEN_UNITS = 128
LEARNING_RATE = 0.001
NETWORK_UPDATE_FREQUENCY = 100

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


class NoisyLayer(nn.Module):
    def __init__(self, input_features, output_features, sigma=0.5):
        super(NoisyLayer).__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.sigma = sigma
        self.bound = input_features**(-0.5)

        self.mu_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(output_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))
        self.sigma_weight = nn.Parameter(torch.FloatTensor(output_features, input_features))

        self.epsilon_input = None
        self.epsilon_output = None
        self.register_buffer('epsilon_input', torch.FloatTensor(input_features))
        self.register_buffer('epsilon_output', torch.FloatTensor(output_features))

        self.parameter_initialization()
        self.sample_noise()

    def parameter_initialization(self):
        self.mu_bias.data.uniform_(-self.bound, self.bound)
        self.sigma_bias.data.fill_(self.sigma * self.bound)
        self.mu_weight.data.uniform_(-self.bound, self.bound)
        self.sigma_weight.data.fill_(self.sigma * self.bound)

    def forward(self, x: torch.Tensor, sample_noise: bool = True) -> torch.Tensor:
        if not self.training:
            return F.linear(x, weight=self.mu_weight, bias=self.mu_bias)

        if sample_noise:
            self.sample_noise()

        weight = self.sigma_weight * torch.ger(self.epsilon_output, self.epsilon_input) + self.mu_weight
        bias = self.sigma_bias * self.epsilon_output + self.mu_bias
        return F.linear(x, weight=weight, bias=bias)

    def sample_noise(self):
        self.epsilon_input = self.get_noise_tensor(self.input_features)
        self.epsilon_output = self.get_noise_tensor(self.output_features)

    def get_noise_tensor(self, features: int) -> torch.Tensor:
        noise = torch.FloatTensor(features).uniform_(-self.bound, self.bound)
        return torch.sign(noise) * torch.sqrt(torch.abs(noise))


class DQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc_1 = NoisyLayer(input_features, HIDDEN_UNITS)
        self.fc_2 = NoisyLayer(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc_3 = NoisyLayer(HIDDEN_UNITS, output_features)

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
        self.memory = Memory(memory_capacity)  # TODO add gamma

    def train(self, episodes):
        env = gym.make(ENVIRONMENT)
        self.update_target_network()
        recent_scores = deque(maxlen=100)
        recent_lengths = deque(maxlen=100)
        recent_losses = deque(maxlen=100)

        for episode in range(episodes):
            state = env.reset()
            score, i = 0, 0
            for i in count():
                action = self.agent(state)
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

            recent_lengths.append(i)
            recent_scores.append(score)
            print(self.step, episode, np.mean(recent_scores), np.mean(recent_lengths), np.mean(recent_losses))

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
    trainer = DeepQLearning(agent, target_network, local_network_optimizer, MEMORY_CAPACITY, BATCH_SIZE)
    trainer.train(2000)

