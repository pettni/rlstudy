# Repository guidelines

This is an RL self-study project. The solver code is meant to make each algorithm easy to follow.

- Keep solver implementations close to the equations and training loop. Prefer straightforward, top-to-bottom code over abstractions, helpers, and configuration added for hypothetical reuse.
- Use clear names for states, actions, rewards, returns, and update steps. Comment on a non-obvious algorithmic choice; let ordinary code explain itself.
- Keep reporting, evaluation, CLI parsing, and plotting outside the core solver logic. Solvers should expose only measurements they already compute and their current policy.
- Make the smallest change that solves the request. Preserve existing behavior and style unless the request calls for a change.
- Prefer a few simple end-to-end tests that run a solver on a small environment and check its observable behavior. These should remain useful when the implementation is refactored.
- Add a focused unit test for a subtle numerical rule or a reproduced bug when an end-to-end test would not pin it down. Avoid tests for every field, branch, or implementation detail.
- Run the smallest relevant checks first, then `cargo test` and `cargo fmt --check` when Rust code changes. Report what was run and any limits.
