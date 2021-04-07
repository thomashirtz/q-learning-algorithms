<!-- This is commented out. -->
# 🔬 reinforcement-learning

This repository will contain implementation of reinforcement learning algorithm. The end goal is to have a library of algorithm ready to use, as simple as possible, but not simpler. It is partly inspired by the repository [minimalRL](https://github.com/seungeunrho/minimalRL) but less extreme with the implementations.


## Algorithms

![dueling-dqn-cartpole](images/dueling-dqn-cartpole.gif)

- [x] [Deep Q-Learning](deep_q_learning)
- [x] [Double Deep Q-Learning](double_deep_q_learning)
- [x] [Dueling Deep Q-Learning](dueling_deep_q_learning)
- [x] [Noisy Deep Q-Learning](noisy_networks)
- [x] [Deep Q-Learning with Prioritized Replay](prioritized_experience_replay)
- [x] [Parametrized Deep Q-Learning](parameterized_deep_q_learning)
- [x] [Distributed Deep Q-Learning](distributed_deep_q_learning)
- [ ] Multi-Step Deep Q-Learning
- [ ] Rainbow Q-Learning
- [ ] Ape-X

## Recording the environment

Save an mp4 video of an agent:

```python
env = gym.make(ENVIRONMENT)
env = gym.wrappers.Monitor(env, "./video", force=True)
state = env.reset()
done = False
while not done:
    action = agent.choose_action(state)
    state, _, _, _ = env.step(action)
```
## Requirements

gym  
numpy  
pytorch  

## 📚 References:
[📺 Deepmind/UCL reinforcement learning courses on Youtube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)\
[📄 Deepmind/UCL reinforcement learning website](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)\
[📓 Reinforcement Learning: An Introduction - Richard S. Sutton and Andrew G. Barto](RL%20DeepMind/RLbook2018.pdf)
