# Reinforcement-Learning Algorithm Taxonomy

Algorithms can be compared along these dimensions:

| Dimension | Main choices |
| --- | --- |
| Environment knowledge | Model-based or model-free |
| Learned quantities | Model, V(s), Q(s,a), policy, or actor and critic |
| Data policy | On-policy or off-policy |
| Value target | Monte Carlo, one-step TD, n-step TD, or lambda-return |
| Policy signal | Return, Q-value, TD error, or advantage |
| Data usage | Online transition, episode, trajectory batch, or replay buffer |
| Policy | Deterministic or stochastic |
| Action space | Discrete or continuous |
| Representation | Tabular, linear, or neural |
| Exploration | Epsilon-greedy, policy sampling, entropy, or explicit bonuses |

## Target and policy signal

A target is the value an estimator tries to predict. A policy signal determines
how the policy changes.

For an actor-critic method:

```text
critic target = discounted or bootstrapped return
critic loss   = (V(s) - target)^2

policy signal = advantage
policy loss   = -log pi(a|s) * advantage
```

The advantage is:

```text
A(s,a) = Q(s,a) - V(s)
```

A positive advantage increases the probability of the sampled action. A
negative advantage decreases it.

Value-based methods such as DQN have no separate policy signal. They train Q
toward a target and derive the policy by selecting the action with the largest
Q-value.

## Algorithms

| Algorithm | State space | Action space | Policy relation | Learned quantities | Value target | Policy signal | Data usage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Value iteration | Discrete | Discrete | Model-based | V or Q | Bellman optimality | None; policy derived from values | Full sweeps |
| SARSA | Discrete | Discrete | On-policy | Q | One-step TD using the sampled next action | None; policy derived from Q | Online |
| Approximate TD(n) | Continuous | Discrete | On-policy | Neural Q | n-step TD using the sampled bootstrap action | None; policy derived from Q | Online n-step window |
| Q-learning | Discrete | Discrete | Off-policy | Q | One-step TD using the greedy next action | None; policy derived from Q | Online |
| DQN / DDQN | Continuous | Discrete | Off-policy | Neural Q | Q-learning target; DDQN separates next-action selection from evaluation | None; policy derived from Q | Replay batches |
| DDPG | Continuous | Continuous | Off-policy | Deterministic policy and neural Q | One-step TD using the target policy and target Q | Q gradient through the critic | Replay batches |
| REINFORCE | Continuous | Discrete or continuous | On-policy | Policy | None without a critic | Monte Carlo return | Episodes |
| REINFORCE with baseline | Continuous | Discrete or continuous | On-policy | Policy and V | Monte Carlo or TD critic target | Return minus V(s) | Episodes |
| Actor-critic | Continuous | Discrete or continuous | On-policy | Policy and V | One-step TD | TD error | Online |
| VPG | Continuous | Discrete or continuous | On-policy | Policy and V | Bootstrapped return | GAE advantage | Trajectory batches |
| PPO | Continuous | Discrete or continuous | On-policy | Policy and V | Bootstrapped return | GAE advantage through a clipped probability ratio | Trajectory batches with multiple updates |

## Bias and variance

Monte Carlo targets use observed rewards through the end of a trajectory. They
have low bootstrap bias and often high variance.

TD targets bootstrap from a learned estimate. They usually have lower variance
and introduce bias from that estimate.

n-step returns and lambda-return methods interpolate between those endpoints.
GAE applies the same trade-off to advantage estimation.
