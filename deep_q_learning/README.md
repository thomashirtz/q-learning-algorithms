## Deep D-Learning

This [[algorithm]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) come from a modification of the Q-Learning technique adapted to be coupled with function approximation.
The core of the algorithm is taken from the famous [[tutorial]](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
from pytorch. The comments linked to the code of the tutorial provide a very detailed explanation. However the code is not 
so minimalistic and if the original code is ran, it will not converge because of the hyperparameters choice.

This implementation is more minimalistic, the hyperparameter are well tuned. The environment should be solved in ~500 episodes (~40 000 steps)

## Algorithm

Here is the algorithm taken from the original paper [[Playing Atari with Deep Reinforcement Learning]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

![algorithm](../images/deep_q_learning_algorithm.jpg)