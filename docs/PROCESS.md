# How we operate

This is the operating manual: the workflow, tooling, and collaboration mechanics for this
project. It answers "what do I do and in what order," not "what must every number respect"
(that's `docs/PROJECT_SPEC.md`'s invariants) and not "what's true right now" (that's
`docs/PROJECT_STATE.md`). Read those two after this one, before starting work.

## 0. Start here, every session

1. Read `docs/PROJECT_SPEC.md`'s spec index — is there already an ACTIVE spec for what you're
   about to do? Extend or explicitly supersede it; don't start a third parallel document.
2. Read `docs/PROJECT_STATE.md` — is the thing you're about to fix already fixed, already known
   broken with a reason, or already being worked in a session you'd be racing?
3. Check `git log --oneline -20` on the branch you're on. Multiple sessions (this machine and
   Alpine, sometimes concurrently) touch this repo; the two documents above are the coordination
   mechanism, not tribal memory of who's doing what.

This replaces writing a fresh dated `HANDOFF-*.md` at the end of a session. Update
`docs/PROJECT_STATE.md` in place instead — a dozen handoff files across three weeks is the
concrete shape of the problem this document set exists to stop.

## 1. The work lifecycle

For anything beyond a small, obviously-scoped fix, this project's standing convention (not new,
already the pattern behind every dated file in `docs/superpowers/`) is:

**Brainstorm → Design spec → Implementation plan → Execute → Review → Verify → Promote.**

| Stage | Where it lives | Skill |
|---|---|---|
| Brainstorm | (conversation) | `superpowers:brainstorming` |
| Design spec | `docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md` | (write directly, or via brainstorming's output) |
| Implementation plan | `docs/superpowers/plans/YYYY-MM-DD-<slug>.md` | `superpowers:writing-plans` |
| Execute | code changes | `superpowers:executing-plans` or `superpowers:subagent-driven-development`, with `superpowers:test-driven-development` and `superpowers:systematic-debugging` as needed |
| Review | — | `superpowers:requesting-code-review` / `receiving-code-review` |
| Verify | — | `superpowers:verification-before-completion` — evidence before "done," always |
| Promote | `docs/results/*.csv` + `.provenance.json` sidecar | `scripts/promote_result.py` |
| Close out | branch merge/PR | `superpowers:finishing-a-development-branch` |

**A lighter path is fine for fast diagnostic work** (the kind that produced
`docs/transfer_ladder_protocol.md` and `docs/decisions/2026-08-25-ladder-round.md` — a
design doc and a decisions log without a dated implementation plan alongside them). That's a
legitimate shortcut under time pressure, not a violation — but it still needs the two things the
shortcut is tempted to skip: a decision entry for anything decided without asking (even one
line), and a `docs/PROJECT_STATE.md` update the moment a number changes. What's not fine is
skipping those *and* never writing a spec at all — that's how a reversal like the CoderData one
goes undocumented for two months.

**Promotion is the line between "we ran something" and "it's evidence."** A result without a
`.provenance.json` sidecar is not citable in a write-up, per this project's own standing rule —
see invariant 8 in `PROJECT_SPEC.md`.

## 2. Compute: local vs. Alpine

- **Local (this machine)**: interactive development, doc/spec work, fast tests, anything that
  doesn't need a GPU or the cluster's data. Run `uv run pytest` here before every push.
- **Alpine**: anything needing real compute or the large staged datasets (Tahoe, L1000, GDSC2
  raw). Access is exclusively through `./scripts/alpine/ralpine` — never raw `ssh`/`scp`. That
  boundary is enforced in version-controlled script code (`READ_ONLY` allowlist +
  `reject_metacharacters`), not in a permission string, specifically so it's reviewable.
- **Deployment is git-only**: `git push` → `ralpine update` (git pull --ff-only on the Alpine
  checkout) → `ralpine submit`. Never copy a file to Alpine directly — if Alpine needs a file
  that isn't code, that's a sign it belongs in `data/` with a download/build script, not a
  one-off transfer.
- **Chained jobs**: pass `--dependency=afterok:<jobid>` etc. to `ralpine submit`; the wrapper
  reorders args correctly, but *verify* with `ralpine run scontrol show job <id> | grep
  Dependency` after submitting — a chain that silently drops its dependency runs immediately and
  out of order, and this has actually happened on this project (`docs/HANDOFF-2026-08-26.md`'s
  "options before the script" trap).
- **Verify inputs exist before submitting.** `ralpine ls`/`ralpine run find` the files a job
  needs first. Three separate rung-4 submissions this session failed in the first seconds of
  the job on things a 10-second check would have caught (an unbound sbatch variable, a
  never-migrated raw-data path, a stale reference file) — each cost a full queue-wait cycle to
  discover the hard way.
- **Standing permissions** (this branch, granted 2026-08-26, recorded in `CLAUDE.md`):
  `git commit`/`git push` to `origin`, `ralpine submit`/`cancel`/`update`, all without
  per-action confirmation. Still confirm before force-push, `git reset --hard`, branch deletion,
  or anything touching `upstream` or `main`. A standing grant is scoped to what it says — if a
  new branch or a materially different kind of action comes up, ask.

## 3. Verification discipline

- **Test on synthetic data before spending cluster time.** Build a small synthetic fixture that
  exercises the real code path (not a reimplementation of its logic — see invariant 4), run it
  locally, confirm the shape and sign of the result look right, *then* submit to Alpine. This is
  what caught the rung-2 rewrite's correctness before it ever touched real data this session.
- **Every reported statistic needs a known-answer test.** Plant a signal, require it recovered;
  plant nothing, require null. The test imports the real function.
- **Run the full local suite before every push**: `uv run pytest -q`.
- **`verification-before-completion` applies to claims, not just code**: before saying something
  is fixed, promoted, or done, show the command and its output. "Should work now" is not done.

## 4. Git and collaboration

- **Branches**: `worktree-<topic>`, checked out under `.claude/worktrees/<topic>/` alongside the
  main checkout (`git worktree list` shows all active ones — check for collisions before naming
  a new one).
- **Commit messages state the root cause, not just the change**, written so a future session (or
  another agent) can tell what happened and why without re-deriving it from the diff — this is
  the established convention in this repo's own history, not a preference introduced this
  session.
- **Push to `origin`** (the `lagillenwater` fork). **Never `upstream`** (`greenelab`) without
  being explicitly told. Greene Lab standard beyond this fork: fork-per-contributor, PRs only,
  minimum one lab-member approval, one functional area per PR.
- **Multiple sessions run concurrently** (this branch has provenance sidecars this session found
  pointing at *another machine's* `/private/tmp/` scratch path from earlier the same day). Don't
  assume you're the only session touching this repo — `docs/PROJECT_STATE.md` and recent git log
  are how you find out, not memory of who was told to do what.

## 5. Documentation discipline

Full rules in `docs/PROJECT_SPEC.md`'s "Process for new specs" section; the short version:

- Superseding a doc = a one-line dated banner on the *old* doc pointing to the new one, **and**
  an index-row update in `PROJECT_SPEC.md`, in the same change. Not one or the other.
- Reversing an architecture, method, or data source = a decision entry in `docs/decisions/`,
  even one paragraph, even written late. Late and terse beats never.
- A number change = `docs/PROJECT_STATE.md` updated in the same change that produced it.

## 6. Definition of done

- **A bug fix**: a real test (imports the actual function) passes locally; if it touches a
  promoted number, that result is re-run, re-promoted with a fresh sidecar, and the document
  that reported the old number gets a correction banner — never a silent edit, per the
  aggregate-vs-per-item p-value fixes this session (`docs/l1000_imputation_fidelity.md`'s
  banner is the template).
- **A new capability**: went through brainstorm → spec → plan, or the explicitly-allowed light
  path (§1) with its two non-negotiables (decision entry, state update) actually done.
- **A claim of "complete"**: backed by command output you actually ran, not by what the code
  should do.
