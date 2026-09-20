# rlstudy CLI

Solve a YAML grid world:

```bash
cargo run -r -- solve --grid-world grid_worlds/zhao_5x5.yaml vi --gamma 0.9
```

Solvers are `vi`, `pi`, `tpi`, `mc`, `td`, `dqn`, and `policy-gradient`. Use `--help` for their
options.

```bash
cargo run -r -- solve --grid-world grid_worlds/zhao_3x3.yaml td --episodes 10000 --episode-length 30 --epsilon 0.1 --alpha 0.1
cargo run -r -- solve --grid-world grid_worlds/zhao_3x3.yaml td --variant expected-sarsa
cargo run -r -- solve --grid-world grid_worlds/zhao_3x3.yaml td --variant q-learning
```

Load the selected solver's settings from YAML with the solve-level `--config`
option. Set reporting frequency and evaluation size with solve-level options:

```bash
cargo run -r -- solve --grid-world grid_worlds/zhao_5x5.yaml --config settings/dqn_for_5x5.yaml --evaluation-frequency episode:50 --evaluation-episodes 10 --evaluation-max-steps 100 dqn
```

Run DQN on Gymnasium through the Python environment managed by `uv`:

```bash
uv run cargo run -r -- solve --gym CartPole-v1 dqn
```

## Plot results for tuned solvers

Value iteration on grid world:
```bash
cargo run -r -- solve --json --grid-world grid_worlds/zhao_5x5.yaml vi | uv run scripts/plot_solver.py | icat
```

Grid world solutions also support `scripts/plot_grid_world.py` to visualize the policy and state values.

DQN on 5x5 grid:
```bash
cargo run -r -- solve --evaluation-frequency episode:10 --json --grid-world grid_worlds/zhao_5x5.yaml --config settings/dqn_for_5x5.yaml dqn | uv run scripts/plot_solver.py | icat
```

DQN on CartPole:
```bash
uv run cargo run -r -- solve --evaluation-frequency episode:10 --evaluation-max-steps 500 --json --gym CartPole-v1 --config settings/dqn_for_cartpole.yaml dqn| uv run scripts/plot_solver.py | icat
```

REINFORCE on CartPole:
```bash
uv run cargo run -r -- solve --evaluation-frequency episode:10 --evaluation-max-steps 500 --json --gym CartPole-v1 --config settings/reinforce_for_cartpole.yaml policy-gradient | uv run scripts/plot_solver.py | icat
```
