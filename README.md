<!-- This is commented out. -->
# 🔬 reinforcement_learning

This repo will share content, notes and material to learn reinforcement leanring. 
My first goal is to follow the DeepMind class, try to implement all the demonstration of the classes using Python and share the notes related to the core consepts discussed in the class. Depending on the progress, I may extend this repo to the Sutton Book and work by Chapter. I will also share complementary material in each section.\
I will try to not overwelm the reader with material and boil down the information.

## 📚 References:
[📺 Deepmind/UCL reinforcement learning courses on Youtube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs)\
[📄 Deepmind/UCL reinforcement learning website](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)\
[📓 Reinforcement Learning: An Introduction - Richard S. Sutton and Andrew G. Barto.](RL%20DeepMind/RLbook2018.pdf)\

## [(Deepmind) Reinforcement Learning 1: Introduction to Reinforcement Learning](https://youtu.be/ISk80iLhdfU)

# Principle

TODO ~8m30

What is reinforcement learning ?
"Reinforcement learning (RL) is an area of machine learning concerned with how software agents ought to take actions in an environment in order to maximize some notion of cumulative reward. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning."\
-[Wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning)

You can have a reinforcement learning framework/Problems because of the sequential nature that they possess without necessarely applying reinforcement learning methods/algorithms to solve them.

At each step t:\
* The agent:
    * Receives observation Ot (and reward Rt)
    * Executes action At
* The environment:
    * Receives action At
    * Emits observation Ot+l (and reward Rt+1)

# Reward & Return

**A reward is a scalar feedback signal**

The Return (G) is equal to the Cumulative Reward (R)

![G_t=R_{t+1}+R_{t+2}+R_{t+3}+...](https://render.githubusercontent.com/render/math?math=G_t%3DR_%7Bt%2B1%7D%2BR_%7Bt%2B2%7D%2BR_%7Bt%2B3%7D%2B...)
<!-- G_t=R_{t+1}+R_{t+2}+R_{t+3}+... -->

**Goal ⇒ Maximize the expected total cumulative reward**

In computer science, reward is reward even if it is negative, even though it should be called penalty in theory

Continuing problem, the sum/problem doesn't end

Terminal condition ⇒ When to reset the environment 

- Time limits
- Positive terminals
- Negative terminals

MDP: In a state st you take an action at and then you observe a reward rt+1 and a state st+1

Bandits: action at time t and a reward at the same time t

## Type of Reward

### Positive/Negative

Positive rewards encourage:

- Keep going to accumulate reward
- Avoid terminals unless they yield very high reward
(terminal state yields more single step reward than the
discounted expected reward of continuing the episode.)

Negative rewards encourage:

- Reach a terminal state as quickly as possible to avoid accumulating
penalties.

### Sparse/Shaped

* Sparse reward ⇒ 1 win, 0 loss
* Shaped reward (or shaping) ⇒ offers a smooth gradient of rewards as the agent approach the objective 

Designing reward functions is a hard problem indeed. Generally, sparse reward functions are easier to define (e.g., get +1 if you win the game, else 0). However, sparse rewards also slow down learning because the agent needs to take many actions before getting any reward. This problem is also known as the **credit assignment problem**.
https://www.cs.ubc.ca/~murphyk/Bayes/pomdp.html

## Discount Factor (β or γ)

[Stackexchange - Role of discount factor](https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning)

We need the discount factor gamma, to compute the total cumulative reward given by the sum of whole single step rewards, This means that getting a larger reward now and a smaller reward later is preferred to getting a smaller reward now and a larger reward later.
* If γ=0, the agent will be completely myopic and only learn about actions that produce an immediate reward. 
* If γ=1, the agent will evaluate each of its actions based on the sum total of all of its future rewards

## Additionnal Material

[📺**Writing Great Reward Functions - Bonsai**](https://www.youtube.com/watch?time_continue=128&v=0R3PnJEisqk&feature=emb_logo)

10:22 => Raise to power less than one ⇒ Add some sharpness ⇒ Gradient is sharp towards the goal\
Often best to bound reward between -1 and 1

# Value

**Value is the expected cumulated reward**

Notation: E = Expected

![v(s)=\mathbb{E}\[G_t|S_t=s\] \\ =\mathbb{E}\[R_{t+1}+R_{t+2}+R_{t+3}+... |S_t=s\]](https://render.githubusercontent.com/render/math?math=v(s)%3D%5Cmathbb%7BE%7D%5BG_t%7CS_t%3Ds%5D%20%5C%5C%20%3D%5Cmathbb%7BE%7D%5BR_%7Bt%2B1%7D%2BR_%7Bt%2B2%7D%2BR_%7Bt%2B3%7D%2B...%20%7CS_t%3Ds%5D)
<!-- v(s)=\mathbb{E}[G_t|S_t=s] \\ =\mathbb{E}[R_{t+1}+R_{t+2}+R_{t+3}+... |S_t=s] -->
Also:

![G_t=R_{t+1}+G_{t+1}](https://render.githubusercontent.com/render/math?math=G_t%3DR_%7Bt%2B1%7D%2BG_%7Bt%2B1%7D)
<!-- G_t=R_{t+1}+G_{t+1} -->
V* is the optimal value, obtained using an optimal policy

⇒ Goal is to select actions to maximise **value**

# Action value

**Possible to condition the value on actions**

![q(s,a)=\mathbb{E}\[G_t|S_t=s, A_t=a\] \\ =\mathbb{E}\[R_{t+1}+R_{t+2}+R_{t+3}+... |S_t=s, A_t=a\]](https://render.githubusercontent.com/render/math?math=q(s%2Ca)%3D%5Cmathbb%7BE%7D%5BG_t%7CS_t%3Ds%2C%20A_t%3Da%5D%20%5C%5C%20%3D%5Cmathbb%7BE%7D%5BR_%7Bt%2B1%7D%2BR_%7Bt%2B2%7D%2BR_%7Bt%2B3%7D%2B...%20%7CS_t%3Ds%2C%20A_t%3Da%5D)

<!-- q(s,a)=\mathbb{E}[G_t|S_t=s, A_t=a] \\ =\mathbb{E}[R_{t+1}+R_{t+2}+R_{t+3}+... |S_t=s, A_t=a]$$ -->

# Environment

**Environment state is the environment internal state** (Usually not visible to the agent)

History ⇒ Sequence of observations, actions, rewards

![H_t=O_0,A_0,R_0,...,O_{t-1},A_{t-1},R_t,O_t](https://render.githubusercontent.com/render/math?math=H_t%3DO_0%2CA_0%2CR_0%2C...%2CO_%7Bt-1%7D%2CA_%7Bt-1%7D%2CR_t%2CO_t)
<!-- H_t=O_0,A_0,R_0,...,O_{t-1},A_{t-1},R_t,O_t -->

**History can be used to construct an "Agent State" S_t**

## Fully observable Environments

**Environment completely observable ⇒ the agent = Markov decision Process (MDPs)**

Observation = Environment state (ex: Single-player board game)\
S_t = O_t = environment state\
Markov processes ~ short memory\
The future does not depends on the whole story, but only on the current state

"the future is independent of the past given the present"

## Partially Observable Environments

Environment partially observable where some variables important are not observable\
Dynamic latent variable models ~ Hidden Markov Models

Partial observability: The agent gets partial information

- A robot with camera vision isn't told its absolute location
- A poker playing agent only observes public cards

Now the observation is not Markov\
Formally this is a partially observable Markov decision process (POMDP)\
The environment state can still be Markov, but the agent does not know it
