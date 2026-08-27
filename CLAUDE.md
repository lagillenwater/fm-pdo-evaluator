# Project instructions — fm-pdo-evaluator

## Standing permissions (worktree-modular-harness-core, 2026-08-26)

Lucas has explicitly authorized the following without per-action confirmation, to avoid
burning session budget on approval round-trips during the ladder round:

- `git commit` and `git push` to `origin` (the `lagillenwater` fork), on this branch or any
  branch created for this work. Still never push to `upstream`, never force-push, never
  push to `main` directly — open a PR as usual.
- `./scripts/alpine/ralpine submit <script.sbatch> [args]` — submitting Alpine jobs.
- `./scripts/alpine/ralpine cancel <JOBID>` — cancelling one Alpine job by numeric id (the
  script already restricts this to a single id with no ranges). This covers our own jobs that
  are stale (superseded by a newer submission, or running code a later commit already fixed)
  or running far past their expected wall time. Never another user's job.
- `./scripts/alpine/ralpine update` — `git pull --ff-only` on the Alpine checkout.

This overrides the general default of checking before git commit/push. Still check before
any destructive/irreversible action not listed above (force-push, `git reset --hard`,
deleting branches, etc.) and before touching `upstream` or `main`.
