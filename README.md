# Taxi-v3-RL-model
This project trains and compares three PPO agents on the classic Taxi-v3 environment with varying learning rate

## Taxi-v3 environment
The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations. It consists of a discrete action and observation space
- Sparse rewards (+20 for drop-off, −1 per time step)
- The reward we will get is much down the path (long term reward assignment)
- Stochastic transitions (due to navigation penalties)
- The need for consistent exploration to avoid getting stuck at a local maxima for the objective

## DQN vs PPO

### Exploration
One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the trade-off between **exploration and exploitation**.

DQN uses ε-greedy exploration. This is very _primitive_. It does not model a true stochastic policy as the DQN learns the Q-values and gets the policy greedily from it and does not learn the policy.
- With probability ε → take a random action (exploration)
- With probability 1 − ε → take the best action according to current Q-values (exploitation)
- ε usually decays over time (start high → gradually decrease)

On the other hand PPO outputs truly stochastic policies
It outputs a distribution (Gaussian for continuous action spaces and Categorical for discrete action space) and the entropy of the distribution controls exploration. Entropy bonus in the objective encourages better exploration.

### Stability
DQN uses the target network to compute Q-targets. If the target estimate is wrong then we train on wrong targets and the target errors propagate forward.
It also suffers from **value overestimation** and noise as it blindly selects the highest value of the next state while calculating the target
The **Off-policy learning** makes it sample effecient but more prone to bias and instability.

On the other hand PPO is designed to be stable as it clips the change in agent's policy in the range *1−ϵ to 1+ϵ*. This makes the learning curve much smoother and makes it less sensititve to noise in the environment.

### Advantage Learning vs Q-Value Learning
**Q-value learning** tries to estimate **Q(s, a)** for *each action in each state*. This means the network must learn long-term return, future discounted rewards for each action which is a very difficult estimation task

 Because these Q-targets are bootstrapped and noisy, Q-values often become **unstable and high variance**, overly large, or sensitive to reward scaling. This makes algorithms like DQN harder to train reliably.

 Action selection depends entirely on argmax Q(s,a), making small errors matter a lot.

On the other hand, **Advantage learning** focuses on **relative action quality**, not absolute value. The advantage tells the agent how much better an action is than the baseline value of the state. Because of this subtraction, a lot of variance cancels out, producing cleaner learning signals and more stable gradients. This is why policy-gradient methods like PPO prefer advantages.

```A(s, a) = Q(s, a) − V(s)```

Further a good trade-off between variance and bias is achieved using GAE and advantage normalisation

### Long-Term Credit Assignment

**DQN (TD(0))**
- Only uses the next step to update values
- Long-term reward propagation is very slow
- Taxi-v3 has many −1 penalties before the +20 final reward
- Hard for DQN to learn long navigation sequences

**PPO + GAE**

- Uses multi-step TD errors
- Propagates long-horizon information more effectively
- Learns pickup → navigate → drop-off paths faster
