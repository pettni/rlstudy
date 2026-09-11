# rlstudy CLI

Solve a YAML grid world:

```bash
cargo run --quiet -- solve --gamma 0.9 --system grid_worlds/zhao_5x5.yaml vi
```

Solvers are `vi`, `pi`, `tpi`, `mc`, and `sarsa`. Use `--help` for their
options.

```bash
cargo run --quiet -- solve --system grid_worlds/zhao_3x3.yaml sarsa --episodes 10000 --episode-length 30 --epsilon 0.1 --alpha 0.1
cargo run --quiet -- solve --system grid_worlds/zhao_3x3.yaml sarsa --episodes 10000 --episode-length 30 --epsilon 0.1 --alpha 0.1 --expected
```

## Render

Display state values and the greedy policy with `icat`:

```bash
cargo run --quiet -- solve --gamma 0.9 --system grid_worlds/zhao_5x5.yaml vi | uv run scripts/render_grid_world.py grid_worlds/zhao_5x5.yaml | icat
```

Replace `| icat` with `> grid_world.png` to save the image.

## Grid-world YAML

Coordinates are zero-based `[row, column]` pairs.

```yaml
side: 3
initial_distribution:
  type: fixed
  state: [0, 0]
forbidden_states:
  - [1, 2]
goal_states:
  - [2, 2]
r_boundary: -1
r_forbidden: -10
r_target: 1
r_other: 0
```

Use `type: uniform` without `state` for a uniform initial distribution.

## Trace values

```bash
cargo run --quiet -- trace --system grid_worlds/zhao_5x5.yaml --output trace.csv vi
uv run scripts/plot_values.py trace.csv --state 0 --output state-0.png
```
