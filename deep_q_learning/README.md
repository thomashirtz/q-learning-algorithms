## Deep D-Learning

This [algorithm](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) come from a modification of the Q-Learning technique adapted to be coupled with function approximation.
The core of the algorithm is taken from the famous [tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
from pytorch. The comments linked to the code of the tutorial provide a very detailed explanation. However the code is not 
so minimalistic and if the original code is ran, it will not converge because of the hyperparameters choice.

This implementation is more minimalistic, the hyperparameter are well tuned. The environment should be solved in ~500 episodes (~40 000 steps)

## Algorithm

Here is the algorithm taken from the original paper [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

<img width="750" alt="Deep Q-Learning Algorithm" src="../images/deep_q_learning_algorithm.jpg">

## Code breakdown

### Memory

This part of the algorithm will contain all the last transition that the agent has been through. The transistion are stored into a `namedtuple` container which will contain the state, the cation, the reward, the next state as well as the 'done' indicator.

```
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
```
The memory is composed of a deque of size capacity, when a sample is pushed but the memory is full, it will discard the oldest sample and append the new sample to 
the queue.
When the `sample` method is called, a batch of experience are sampled from the queue. 
In this implementation, instead of returning a list of experience, I prefer to return a single experience object that contains list of state, list of action, etc. The command `Experience(*zip(*experiences))` enable this.

### Deep Q-Network
```
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
```

### Policy

```
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
```

## Results 

![Deep Q-Learning](../images/deep_q_learning.png)
