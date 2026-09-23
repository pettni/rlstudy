# Reinforcement Learning for Robotics

A theory-first, project-driven path from classical reinforcement learning to modern deep RL for robotics.

## Goal

Develop a solid working understanding of the RL options for improving pretrained robot policies from execution data beyond expert demonstrations. By the end, I should be able to:

- derive and explain the main ideas behind dynamic programming, Monte Carlo, temporal-difference learning, policy gradients, and actor–critic methods;
- explain the practical differences between on-policy, off-policy, offline, and iterated offline learning;
- implement and test representative tabular algorithms;
- implement and debug PPO and Soft Actor-Critic (SAC);
- learn value functions from mixed demonstrations, autonomous trials, failures, and corrections;
- compare direct online fine-tuning, off-policy learning with prior data, offline RL, regularized policy extraction, and correction-based learning;
- estimate advantages from execution data and use them for policy improvement;
- evaluate success rate, throughput, failure rate, and robustness across iterations and random seeds;
- read current robot-learning systems and identify which established RL components they combine.

## Working approach

- Read Shiyu Zhao's *Mathematical Foundations of Reinforcement Learning* end-to-end as the main curriculum.
- Implement a small algorithm or experiment for each conceptual jump in Zhao using `~/coding/rlstudy`. Implement enough to expose the important details; do not reproduce every variant.
- Use Rust for tabular and neural implementations. Use Burn for neural networks and custom training loops.
- Keep Python behind small environment bridges when a simulator has a Python-first API. Rust should own the algorithm, models, replay or rollout data, optimization, evaluation, and logging.
- Use Sutton and Barto for alternate explanations, exercises, and topics that need more depth.
- Use Bertsekas's *Reinforcement Learning and Optimal Control* selectively for dynamic programming, approximation, optimal control, and model-based perspectives.
- Transition to [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) after finishing Zhao. Use its explanations and reference implementations to guide equivalent Rust/Burn implementations.
- Treat world models, behavior cloning, and VLA architecture as existing background. Study them only where they interact directly with RL from execution data.
- Use recent robot-learning systems as case studies, not as the organizing structure of the curriculum.

## Current status

As of 2026-09-23:

- [x] Started reading Zhao end-to-end.
- [x] Ordered Zhao and Sutton & Barto.
- [x] Have Bertsekas's *Reinforcement Learning and Optimal Control*.
- [x] Started `~/coding/rlstudy`.
- [x] Implemented value iteration, policy iteration, and truncated policy iteration.
- [x] Implemented epsilon-greedy Monte Carlo control.
- [x] Completed the Chapter 6 stochastic-approximation experiment.
- [x] Implemented Sarsa, Expected Sarsa, and Zhao's on-policy Q-learning.
- [x] Read and derived tabular n-step Sarsa; plan to implement the neural semi-gradient version separately.
- [x] Evaluated an agent-built Burn DQN prototype with a PyO3 Gymnasium bridge.
- [x] Decided to continue neural RL in Rust/Burn based on the prototype.
- [x] Refocused the final target on comparing RL options for learning from execution data.
- [x] Added Grid World tests for the implemented algorithms.
- [x] Personally implemented DQN on CartPole.
- [x] Understood linear value-function approximation; implementation intentionally skipped because the shared approximation concept is clear and DNN experience directly supports the project goal.
- [x] Implemented REINFORCE and a learned state-value baseline in the `Reinforce` solver.
- [x] Compared Monte Carlo and one-step TD actor/critic targets on CartPole.
- [x] Implemented online one-step actor–critic with per-transition updates; CartPole performance remains unresolved.
- [x] Implemented episodic semi-gradient n-step Sarsa with neural Q-function approximation.
- [x] Implemented VPG with generalized advantage estimation and bootstrapped value targets.
- [x] Standardized DQN, REINFORCE, and VPG transition storage around a shared transition buffer.
- [x] Wrote an algorithm taxonomy covering policy relation, targets, policy signals, data usage, and bias–variance tradeoffs.

The Zhao implementation checkpoints, including neural n-step Sarsa, are complete. VPG with GAE solves CartPole, providing the bridge into Spinning Up. Online one-step actor–critic remains unreliable on CartPole. Next, work through PPO and implement it on a small continuous-control environment.

## Project

Compare ways to improve a pretrained simulated manipulation policy using its own execution data.

Start with state observations and Cartesian action commands in MuJoCo or ManiSkill. Use a scripted controller or an existing policy to provide enough demonstrations for an initial behavior-cloned policy; building the demonstrator and studying behavior cloning are supporting work, not curriculum goals.

The project should:

1. collect autonomous successes and failures from the current policy;
2. optionally collect corrective interventions during failed executions;
3. apply two policy-improvement methods to comparable data;
4. deploy each improved policy and collect another iteration;
5. explain which assumptions made each method work or fail.

Candidate methods include:

- direct on-policy or off-policy fine-tuning;
- off-policy learning that balances prior and online data, such as RLPD;
- value-based offline RL, such as IQL;
- advantage-weighted regression;
- advantage-conditioned policy extraction;
- residual RL around the pretrained policy.

Final deliverables:

- a reproducible collection and policy-improvement loop;
- a comparison of two RL approaches against the initial policy and behavior cloning on all aggregated data;
- success rate, task throughput, and failure rate over at least three seeds;
- one ablation on corrections, prior-data retention, regularization, or data composition;
- a short account of which method and execution data produced the improvement.

## Phase 1: Complete Zhao and the classical foundation

Read all ten chapters in order. Pair each major concept with a bounded teaching implementation.

### Implementation checkpoints

- [x] **Chapter 4:** value, policy, and truncated policy iteration.
- [x] **Chapter 5:** Monte Carlo control.
- [x] **Chapter 6:** incremental sample means and one stochastic-approximation convergence experiment.
- [x] **Chapter 7:** one-step Sarsa, Expected Sarsa, and Zhao's on-policy Q-learning.
- [x] **Chapter 7:** derive and understand n-step Sarsa; implementation intentionally skipped.
- [x] **Chapter 8:** understand linear value-function approximation; implementation intentionally skipped in favor of direct DNN implementation.
- [x] **Chapter 8:** personally implement a minimal Burn DQN on CartPole, using the prototype only as a reference after making an independent attempt.
- [x] **Chapter 9:** REINFORCE, then add a learned baseline.
- [x] **Chapter 10:** implement and test online one-step actor–critic with per-transition updates.
- [x] **Sutton and Barto §10.2:** implement episodic semi-gradient n-step Sarsa separately from DQN.
- [ ] Compare one-step and n-step approximate TD learning behavior across several seeds.

Reuse Grid World through the tabular methods and compare learned values or policies with the dynamic-programming solution. Use one small benchmark for the deep methods so network and optimizer behavior can be studied without robotics complexity.

Each implementation should have:

- one environment and one clear algorithm;
- deterministic unit tests where practical;
- a fixed seed for learning tests;
- plots only when convergence behavior matters;
- no framework abstractions beyond what the next implementation already requires.

### Completion checkpoint

Before moving on, be able to explain:

- Bellman expectation and optimality equations;
- sampling versus bootstrapping;
- Monte Carlo versus temporal-difference learning;
- on-policy versus off-policy learning;
- stochastic approximation and the role of step-size schedules;
- value-function approximation and the instability introduced by bootstrapping;
- the policy-gradient objective, baselines, and actor–critic decomposition.

Before starting Spinning Up, read these short Sutton and Barto sections:

- [x] §6.7: maximization bias and Double Learning.
- [x] §§11.1–11.3: off-policy divergence and the deadly triad.

## Phase 2: Transition to deep RL with Spinning Up

Start this phase after Zhao Chapter 10. Use [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) as the bridge from the book's derivations to practical deep RL, translating the relevant implementations into Rust/Burn.

- [x] Read the Spinning Up introduction and key concepts.
- [x] Read Sutton and Barto §§12.1–12.3 on λ-returns before studying PPO and generalized advantage estimation.
- [x] Work through the VPG explanation and implement it in Rust/Burn.
- [ ] Work through the PPO, TD3, and SAC explanations.
- [x] Compare the Zhao Rust/Burn REINFORCE and actor–critic implementations with Spinning Up's VPG formulation.
- [ ] Personally implement PPO with generalized advantage estimation on one small continuous-control environment.
- [ ] Implement a squashed Gaussian actor, replay buffer, twin critics, target networks, and entropy-temperature update for SAC in Rust/Burn.
- [ ] Train SAC on a simple Gymnasium continuous-control environment through a small PyO3 bridge.
- [ ] Compare learning curves with a readable reference implementation such as CleanRL or Stable-Baselines3.
- [ ] Run at least three seeds and inspect failed runs.
- [ ] Read the TRPO derivation only after the rest of the curriculum if natural gradients remain interesting; no implementation is required.

Checkpoint: explain policy-gradient estimation, generalized advantage estimation, PPO clipping, entropy, twin critics, target updates, action scaling, the reparameterization trick, replay data, and the main tradeoffs between PPO and SAC.

## Phase 3: Map the execution-data options

Connect actor–critic methods to the main ways execution data is used after behavior cloning.

- [ ] Study dataset-policy mismatch, extrapolation error, behavior regularization, and why repeated optimization on fixed data differs from online SAC.
- [ ] Derive KL-regularized policy improvement and its connection to advantage-weighted regression (AWR).
- [ ] Implement AWR on a fixed dataset containing both competent and failed executions.
- [ ] Study IQL as an offline-RL route that avoids evaluating unseen actions during value learning.
- [ ] Study RLPD as an off-policy route that mixes prior demonstrations with new online experience.
- [ ] Study direct PPO or SAC fine-tuning and identify when its data requirements are acceptable.
- [ ] Study value- or advantage-conditioned policy extraction, using RECAP as one recent example.
- [ ] Study residual RL as a way to preserve a competent base policy while learning corrections.
- [ ] Represent interventions explicitly and understand their connection to human-gated DAgger.
- [ ] Build a decision table covering data access, exploration, conservatism, policy architecture, and expected failure modes for each route.

Optional case study after PPO: examine [GRPO](https://arxiv.org/abs/2402.03300) for grouped simulator rollouts, using [TGRPO](https://arxiv.org/abs/2506.08440) as a robotics example. Compare its rollout cost and group-relative advantage signal with PPO before considering it for the project.

Checkpoint: given a pretrained policy, dataset, simulator access, and correction budget, choose a plausible improvement method and explain the tradeoffs.

## Phase 4: Compare two improvement routes

This is an algorithm comparison, not an attempt to reproduce a large robot-learning system.

- [ ] Create a reaching, pushing, or insertion environment with explicit sparse success and time-to-completion signals.
- [ ] Establish a behavior-cloned starting policy from scripted demonstrations.
- [ ] Collect complete autonomous episodes, including failures, and retain the behavior-policy or iteration identity.
- [ ] Add scripted or human-triggered corrective interventions for recoverable failures.
- [ ] Aggregate demonstrations, prior iterations, autonomous trials, and corrections without discarding negative examples.
- [ ] Choose two materially different routes from Phase 3.
- [ ] Give both methods comparable data and evaluation budgets.
- [ ] Compare against the initial policy and behavior cloning on all aggregated data.
- [ ] Collect one additional iteration with each improved policy.
- [ ] Ablate one assumption that differs between the methods.

Checkpoint: demonstrate an improvement attributable to execution data and explain why one route was better suited to the setting.

## Phase 5: Synthesize

- [ ] Write a brief project report covering the task formulation, algorithm, baselines, results, ablation, failures, and next experiment.
- [ ] Record which problems came from optimization, exploration, reward design, control authority, or environment design.
- [ ] Map each implementation to the broader families of online RL, off-policy RL with prior data, offline RL, and regularized policy extraction.
- [ ] Decide which route is most promising for the actual robot-learning setting of interest.

## Core resources

Use these in the roles below, not as a parallel reading backlog.

1. [Mathematical Foundations of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) — primary end-to-end curriculum.
2. [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) — transition from classical RL to practical policy gradients and actor–critic methods.
3. Sutton and Barto, *Reinforcement Learning: An Introduction* — targeted reference and exercises.
4. Bertsekas, *Reinforcement Learning and Optimal Control* — targeted reference for dynamic programming, approximation, and control.
5. [Berkeley CS285: Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) — lectures for deep RL topics that need another treatment.
6. [Burn](https://burn.dev/) — neural-network implementation framework in Rust.
7. [`burn-dqn-prototype`](burn-dqn-prototype/PROTOTYPE_REPORT.md) — validated pattern and recorded framework friction.
8. [CleanRL](https://github.com/vwxyzjn/cleanrl) — auditable reference implementations.
9. [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — comparison baseline.
10. [ManiSkill](https://maniskill.readthedocs.io/) or [MuJoCo](https://mujoco.readthedocs.io/) — robotics simulation.

## Papers to read only when needed

- **Soft Actor-Critic Algorithms and Applications** — the main off-policy actor–critic reference.
- **Proximal Policy Optimization Algorithms** — the principal on-policy comparison.
- **Addressing Function Approximation Error in Actor-Critic Methods** — twin critics and target-policy ideas.
- **Advantage-Weighted Regression** — the bridge from learned advantages to regularized policy extraction.
- **Offline Reinforcement Learning with Implicit Q-Learning** — an offline-RL route.
- **Reinforcement Learning with Prior Data (RLPD)** — an off-policy route for mixing demonstrations with online experience.
- **Residual Reinforcement Learning for Robot Control** — a route that learns around an existing controller or policy.
- **π*0.6: a VLA That Learns From Experience** — one case study combining execution data, corrections, value learning, and policy extraction.

## Working rules

- Finish Zhao before starting another broad curriculum.
- Use Sutton and Bertsekas to answer specific questions, not as simultaneous cover-to-cover reading.
- Implement one representative algorithm for each conceptual jump in Zhao.
- Count only personal implementations toward learning checkpoints; agent-built prototypes may validate tooling or serve as later references.
- Keep learning implementations small enough to understand top-to-bottom.
- Keep PPO and SAC because they supply the policy-gradient, critic, replay, and regularization concepts needed to evaluate newer methods.
- Do not add a separate world-model, VLA-pretraining, or behavior-cloning curriculum.
- Organize the later curriculum around algorithmic choices and assumptions, not a particular paper or system.
- Treat TRPO as optional historical depth after the execution-data project.
- Use Rust/Burn for neural RL and custom training loops.
- Keep Python limited to environment or simulator integration.
- Reassess the language decision only if robotics integration creates substantial recurring friction.
- Start with state observations; add vision only after control learning works.
- Keep low-level stabilization outside the learned policy.
- Use explicit train and evaluation environments.
- Fix evaluation seeds, but train with varied seeds.
- Compare against simple baselines before tuning the RL algorithm.
- Log success rate in addition to reward.
- Save videos of evaluation episodes.
- Treat reward changes as hypothesis-driven experiments.
- Prefer one understood algorithm over a broad algorithm survey.

## Progress log

| Date | Milestone or experiment | Result | Next action |
|---|---|---|---|
| 2026-09-11 | Began Zhao and implemented the first algorithms in `rlstudy` | Value, policy, and truncated policy iteration plus Monte Carlo control pass Grid World tests | Implement the Chapter 6 stochastic-approximation experiment |
| 2026-09-11 | Completed the Chapter 6 stochastic-approximation experiment | Step-size schedules showed the expected convergence, freezing, and tracking behavior under a changing target | Continue Zhao and implement TD(0) prediction |
| 2026-09-12 | Completed Zhao Chapter 7 | Implemented one-step Sarsa, Expected Sarsa, and on-policy Q-learning; understood n-step Sarsa without implementing it | Continue to Chapter 8 and implement linear value-function approximation |
| 2026-09-12 | Evaluated an agent-built Burn DQN prototype | Burn DQN reached a 297.4 greedy mean return versus a 27.6 random baseline; this validated Rust/Burn but did not count as a learning checkpoint | Continue Chapter 8 with linear value-function approximation, then implement DQN personally |
| 2026-09-12 | Refocused the curriculum on learning from execution data | PPO and SAC remain foundations; the later phases compare online, off-policy-with-prior-data, offline, and policy-extraction routes | Continue Zhao Chapter 8 |
| 2026-09-19 | Accepted DQN as the Chapter 8 implementation checkpoint | Linear approximation is understood; a separate implementation would add little toward the project goal | Establish REINFORCE learning behavior, add a learned baseline, then implement one-step actor–critic |
| 2026-09-20 | Implemented REINFORCE and a learned baseline in Rust/Burn | The solver now supports Monte Carlo and one-step TD actor/critic targets and reports training and evaluation metrics | Compare update strategies on CartPole |
| 2026-09-20 | Ablated CartPole policy-gradient settings | Monte Carlo actor and baseline reached a 496.5 mean sampled evaluation return across 10 seeds with the learning rate reduced after episode 250; episode-batch TD actor updates remained weak | Test online TD(0) updates after each transition |
| 2026-09-20 | Implemented approximate TD(n) | Added episodic semi-gradient n-step Sarsa with a neural Q function and a MountainCar configuration | Continue into practical policy-gradient methods |
| 2026-09-23 | Implemented VPG in Rust/Burn | Added trajectory batches, GAE, bootstrapped value targets, and CartPole settings that reached a 494.7 mean evaluation return with seed 42 | Work through PPO and continuous policies |
| 2026-09-23 | Standardized transition storage and documented the algorithm taxonomy | DQN, REINFORCE, and VPG now reuse the transition buffer; `TAXONOMY.md` records the main dimensions distinguishing the implemented algorithms | Implement PPO on a small continuous-control environment |
