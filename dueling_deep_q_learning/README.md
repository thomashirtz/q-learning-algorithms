## Double Deep D-Learning

The [[Dueling Q-Learning]](https://arxiv.org/pdf/1511.06581.pdf) is a very subtle modification of the archtecture of the 
Deep Q-Learning. When predicting Q-values, the network will, instead of solely predicting the Q-values, predict the advantage
value at the same time as the state value.

## Implementation

There is only two main difference to change a Deep Q-Learning algorithm into a double is to change the way to compute the expected value:

Deep Q-Learning:
```
class DQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc_1 = nn.Linear(input_features, HIDDEN_UNITS)
        self.fc_2 = nn.Linear(HIDDEN_UNITS, output_features)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        return self.fc_2(x)
```

Dueling Deep Q-Learning (Average):
```
class DuelingDQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc = nn.Linear(input_features, HIDDEN_UNITS)
        self.fc_advantage = nn.Linear(HIDDEN_UNITS, output_features)
        self.fc_value = nn.Linear(HIDDEN_UNITS, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
```

Dueling Deep Q-Learning (Max):
```
class DuelingDQN(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.fc = nn.Linear(input_features, HIDDEN_UNITS)
        self.fc_advantage = nn.Linear(HIDDEN_UNITS, output_features)
        self.fc_value = nn.Linear(HIDDEN_UNITS, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        return value + (advantage - advantage-states.max(dim=1, keepdim=True)[0])
```

The original paper had better results with the "Average" version, they therefore recommended to
 use this version rather than using the "Max" implementation.
 
 *The networks has been shortened for clarity
