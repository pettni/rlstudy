# rlstudy CLI

Solve a YAML grid world:

```bash
cargo run --quiet -- solve --grid-world grid_worlds/zhao_5x5.yaml vi --gamma 0.9
```

Solvers are `vi`, `pi`, `tpi`, `mc`, `td`, and `dqn`. Use `--help` for their
options.

```bash
cargo run --quiet -- solve --grid-world grid_worlds/zhao_3x3.yaml td --episodes 10000 --episode-length 30 --epsilon 0.1 --alpha 0.1
cargo run --quiet -- solve --grid-world grid_worlds/zhao_3x3.yaml td --variant expected-sarsa
cargo run --quiet -- solve --grid-world grid_worlds/zhao_3x3.yaml td --variant q-learning
```

Load the selected solver's settings from YAML with the solve-level `--config`
option:

```bash
cargo run --release -- solve --grid-world grid_worlds/zhao_5x5.yaml --config settings/dqn_for_5x5.yaml dqn
```

Run DQN on Gymnasium through the Python environment managed by `uv`:

```bash
uv run cargo run --release -- solve --gym CartPole-v1 dqn
```

## Plot results

Plot a grid-world policy and state values:

```bash
cargo run --release -- solve --json --grid-world grid_worlds/zhao_5x5.yaml vi | uv run scripts/plot_grid_world.py > grid_world.png
```

DQN grid-world policy:

```bash
cargo run --release -- solve --json --grid-world grid_worlds/zhao_5x5.yaml --config settings/dqn_for_5x5.yaml dqn | uv run scripts/plot_grid_world.py > grid_world_dqn.png
```

DQN training metrics:

```bash
cargo run --release -- solve --json --grid-world grid_worlds/zhao_5x5.yaml --config settings/dqn_for_5x5.yaml dqn | uv run scripts/plot_solver.py > training.png
```

CartPole training metrics:

```bash
uv run cargo run --release -- solve --json --gym CartPole-v1 dqn | uv run scripts/plot_solver.py > cartpole_training.png
```
