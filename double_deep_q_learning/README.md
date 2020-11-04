## Double Deep D-Learning

The [[Double Deep Q-Learning]](https://arxiv.org/pdf/1509.06461v3.pdf) is a modification from the Deep Q-Learning Algorithm. To increase stability, 
when the learning step is predicting the Q-values to choose the "next step", an older and fixed network is used.
This older Network is called "target network". This enable to not directly bootstrap the optimization of the local network with itself, 
resulting in more stability.

## Implementation

There is only two main difference to change a Deep Q-Learning algorithm into a double is to change the way to compute the expected value:

Deep Q-Learning:
```
q_values = self.agent.local_network(state)
next_q_values = self.target_network(next_state)

q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
next_q_value = next_q_values.max(1)[0]
expected_q_value = reward + GAMMA * next_q_value * (1 - done)
```

Double Deep Q-Learning:
```
q_values = self.agent.local_network(state)
q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

next_action = self.agent.local_network(next_state).max(1)[1]
next_q_values = self.target_network(next_state)
next_q_value = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)

expected_q_value = reward + GAMMA * next_q_value * (1 - done)
```

It is also needed to add a function to dopy the weights from the local network to the target network:
```
def update_target_network(self):
    self.target_network.load_state_dict(self.agent.local_network.state_dict())
```
And call this function from time to time:
```
if self.step % NETWORK_UPDATE_FREQUENCY == 0:
    self.update_target_network()
```

Although this technique grant a lot of stability to the network, the update frequency is crucial. Therefore this hyperparameter
need to be carefully tuned. It therefore add some complexity to the tuning of this algorithm.

## Notes

There is two distinct Double Deep Q-Learning paper. One called [[Double Q-Learning]](https://papers.nips.cc/paper/3964-double-q-learning.pdf) published in NIPS in 2010 and one called 
[[Deep Reinforcement Learning with Double Q-learning]](https://arxiv.org/pdf/1509.06461v3.pdf) Published in the AAAI Proceedings in 2015 (and Arxiv). 
While the first one put the basis of the Double Q-Learning, the second one is the technique that most of the people are currently using.

