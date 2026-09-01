# How we operate

**As of** 2026-08-31.

This is the operating manual: the workflow, tooling, and collaboration mechanics for this project.
It answers "what do I do and in what order," not "what must every number respect" (that's `docs/SPEC.md`'s project rules) and not "what's true right now" (that's `docs/STATE.md`).
Read those two after this one, before starting work.

The practice is written down first so those tools get built to it, rather than the practice being reverse-engineered from whatever the tools happened to do.

## 0. Start here, every session

1. Read `docs/SPEC.md`'s spec tree — is there already an OPEN branch covering what you're about to do?
   Extend that task's `docs/tasks/<task-slug>/design.md` or explicitly supersede it; don't open a second task on the same subject.
2. Read `docs/STATE.md` — is the thing you're about to fix already fixed, already known broken with a reason, or already being worked in a session?
3. Check `git log --oneline -20` on the branch you're on.
   Multiple sessions (this machine and the HPC (in this case Alpine), sometimes concurrently) touch this repo; the two documents above are the coordination mechanism, not tribal memory of who's doing what.


## 1. The work lifecycle

**Brainstorm → Design spec → Implementation plan → Execute → Review → Verify → Audit → Promote → Summarise → Close out.**

The first arrow is a gate: brainstorming ends with the design presented and explicitly approved.
Until that approval nothing is edited and nothing is implemented — a revision request reopens the design, it does not approve it.

**Work is organized by task, not by date.**
Every task owns one directory, `docs/tasks/<task-slug>/`, named for what the task does with no date prefix.
The slug is the task's identity: it is what `SPEC.md`'s spec tree keys on and what `STATE.md` links to.
Dates live in git and in each document's own header line.

| Stage | Where it lives | Skill |
|---|---|---|
| Brainstorm | (conversation) | `superpowers:brainstorming` |
| Design spec | `docs/tasks/<task-slug>/design.md` | (write directly, or via brainstorming's output) |
| Implementation plan | `docs/tasks/<task-slug>/plan.md` — interfaces, invariants, ordered steps, and expected test outcomes; never embedded full source. Code exists once, in the repository: a plan that carries the code is a second copy, and a second copy drifts (five of rung 0's audit findings were plan-copy vs shipped-copy divergences) | `superpowers:writing-plans` |
| Execute | code changes | `superpowers:executing-plans` or `superpowers:subagent-driven-development`, with `superpowers:test-driven-development` and `superpowers:systematic-debugging` as needed |
| Review | `docs/tasks/<task-slug>/review.md` — each finding and what was done about it | `superpowers:requesting-code-review` / `receiving-code-review` |
| Verify | `docs/tasks/<task-slug>/verification.md` — the commands run, their output, and pointers to everything the run produced: tables, figures, logs | `superpowers:verification-before-completion` — evidence before "done," always |
| Audit | `docs/tasks/<task-slug>/audit.md` — the design read as numbered, checkable claims, each verified against the landed tree with evidence, every departure classified aligned / recorded / drift; then the fix wave's dispositions; then a **re-audit by a fresh reader** confirming every drift item fixed or recorded. The audit is not passed until the re-audit says so, and it passes **before promotion**, so what gets promoted is what the documents describe | (clause-by-clause read of `design.md` vs the tree; fresh readers for both passes) |
| Promote | `results/<task-slug>/<result>.csv` + a `<result>.provenance.json` record beside it — only the subset the project will cite, copied out of the task folder and made citable. The record carries the result's own checksum, and promotion refuses when the two copies differ | `scripts/promote_result.py` |
| Summarise | `docs/tasks/<task-slug>/summary.md` — the finding in plain language: the hypothesis, the evidence (summary table, control results, figure), the conclusions, and the scripts the task touched. Written for the lab member reviewing the pull request, in statistical and biological terms | (write directly) |
| Close out | branch merge/PR. The pull request opens as a **draft**; the task's documents — `verification.md` especially — get one more human review before it is marked ready. Its description separates the **review surface** (the files needing human eyes) from **generated evidence** (bulk artifacts committed so a reviewer can re-derive results without the cluster), and generated files are marked `linguist-generated` in `.gitattributes` so they collapse in the diff view | `superpowers:finishing-a-development-branch` |

**The lifecycle runs per task and converges; it is not re-entered per run.**
Work added after a stage has passed ripples downstream incrementally — a scoped review of the new commits, an amended `verification.md`, an audit delta over the new surface with the same fresh-reader confirmation, an amended `summary.md` — never a restart of the whole cycle.
Promotion re-triggers only when a promoted claim changes, and that is §6's correction path (re-run, re-promote with fresh provenance, correction banner), not an amendment.
This holds both failure modes off: infinite full-cycle loops on every late addition, and late additions slipping in unreviewed after the gates.

**The ripple is capped.**
Documentary findings batch into a single fix wave; the re-audit re-checks only the items verdicted drift; there is no confirmation pass of a confirmation pass.
Mechanical claims — counts, tallies, whether a cross-reference resolves — are checked by script, never by a fresh reader: agents for judgment, scripts for arithmetic.
Rung 0's audit deltas kept finding documentary defects introduced by the previous fix wave, and one pass was spent reconciling the audit's own clause counting; the cap is what stops prose maintenance from generating its own findings without bound.

Three rules keep the task folder from becoming an orphan:

1. **Create the folder when you write the design spec**, and in that same change add the task's branch to `SPEC.md`'s spec tree, under its rung or under cross-cutting.
   A task folder outside the tree is invisible to the next session, which is the failure mode this structure exists to prevent.
2. **One task, one folder.**
   Extending existing work means editing that task's `design.md`, not starting `<something>-v2`.
   If the scope genuinely changes into different work, open a new task and supersede the old one explicitly (§5).
3. **`STATE.md` links back to the folder.**
   Every status entry names its task's `design.md`, the commits that implemented it, and the outputs it produced (§5) — that link is what makes a number traceable to the intent behind it.

**Decision lineage lives in `decisions.md`, at two levels.**
A dated entry for a change to a task's design or plan goes to that task's `docs/tasks/<task-slug>/decisions.md`, labeled by the document it amends; a dated entry for a change to a project document (this manual, `docs/SPEC.md`) goes to `docs/decisions.md`.
The amended document itself carries only the current position plus a one-line pointer to its lineage — history appended at the foot of a document readers open for current guidance is how the foot outgrows the document.
Every task folder gets a `decisions.md`; SPEC project rule 2 is the binding form.
Decisions that cut across tasks are linked to the `design.md` or other evidence in `docs/tasks/<other-task-slug>/` that drove the change relevant to the current task.


**Promotion is the line between "we ran something" and "it's evidence."**
A result without a `.provenance.json` provenance record is not citable in a write-up, per this project's own standing rule — see project rule 1 in `SPEC.md`.

## 2. Compute: local vs. Alpine

- **Local (this machine)**: interactive development, doc/spec work.
  Run `uv run pytest` here before every push.
- **Alpine**: anything needing real compute.
  Access is exclusively through `./scripts/alpine/ralpine` — never raw `ssh`/`scp`.
  That boundary is enforced in version-controlled script code (`READ_ONLY` allowlist + `reject_metacharacters`), not in a permission string, specifically so it's reviewable.
- **Deployment is git-only**: `git push` → `ralpine update` (git pull --ff-only on the Alpine checkout) → `ralpine submit`.
  Never copy a file to Alpine directly — if Alpine needs a file that isn't code, that's a sign it belongs in `data/` with a download/build script, not a one-off transfer.
- **Chained jobs**: pass `--dependency=afterok:<jobid>` etc. to `ralpine submit`; the wrapper reorders args correctly, but *verify* with `ralpine jobinfo <id> | grep Dependency` after submitting — a chain that silently drops its dependency runs immediately and out of order.
- **Verify inputs exist before submitting.**
  `ralpine ls`/`ralpine du` the files a job needs first.
- **Pull the job's log into `results/<task-slug>/` when it finishes.**
  A log left on the cluster is unreadable to anyone reviewing the pull request, and is gone when the scratch space is cleared.
  It is the record of what the run actually did — the resolved arguments, the warnings, the wall time — so it belongs beside the result it produced.
- **Standing permissions** : `git commit`/`git push` to `origin` on the current task's branch, effective once that task's design is approved; `ralpine submit`/`cancel`/`update` — all without per-action confirmation.
  `submit` is the working default — an agent stages the run and submits it.
  `cancel` covers this project's own jobs that are stale (superseded by a newer submission, or running code that a later commit has already fixed) or running far past their expected wall time; never another user's job, and never a whole array range, which `ralpine cancel` refuses structurally by taking exactly one numeric id.
  Still confirm before force-push, `git reset --hard`, branch deletion, or anything touching `upstream` or `main`.
  A standing grant is scoped to what it says — if a new branch or a materially different kind of action comes up, ask.

## 3. Verification discipline

- **Test on synthetic data before spending cluster time.**
  Build a small synthetic fixture that exercises the real code path (not a reimplementation of its logic), run it locally, confirm the shape and sign of the result look right, *then* submit to Alpine.
- **Every measurement step carries a positive and a negative control** — project rule 4 in `docs/SPEC.md` is the binding form.
  The positive control plants a known signal and requires the real, shipped code to recover it; the negative control feeds signal-free or mismatched data and requires null.
  Both are declared per step in the task's `design.md` and implemented as known-answer tests that import the real function.
- **State the detection power of every reported comparison.**
  Every promoted comparison carries its minimum detectable effect at the declared α and power, computed from the same null bootstrap as its p-value, and positive-control plants are placed relative to it.
  A null result without its MDE cannot be told apart from an underpowered experiment — the distinction rung 5's small cohort will turn on.
- **Run the full local suite before every push**: `uv run pytest -q`.
- **Run the lint and type gates over tracked files, not over hand-picked directories.**
  `git ls-files '*.py' '*.ipynb' | xargs uv run ruff check`, the same with `ruff format --check`, then `uv run pyright`.
  Continuous integration runs `ruff check .` on a clean checkout, which lints notebook code cells too; a local gate scoped to source directories cannot see a violation in a committed notebook, and the branch went red for a day on exactly that gap.
  Selecting tracked files keeps the untracked working-tree strays out (the reason the scoping existed) without hiding anything continuous integration will see.
- **Run the project-rule tests for what you touched.**
  `uv run pytest tests/test_project_rules.py` every time — they may catch a violation the task never intended — plus `-m` for the steps named in the task's `design.md` header, e.g. `uv run pytest -m "step_split or step_score"`.
  The step-to-rule-to-test mapping sits under each rule in `docs/SPEC.md`, and each step's marker is registered in `pyproject.toml` alongside the test that uses it, so the selection is the same whichever rung or dataset the task is about.
  An exemption in that run is a recorded gap, not a pass: if the task closed one, the test starts passing unexpectedly and the exemption comes out in the same change.
- **`verification-before-completion` applies to claims, not just code**: before saying something is fixed, promoted, or done, show the command and its output.
  "Should work now" is not done.
- **A verification check that is cheap to compute is worth running.**
  When an assumption or a claim can be tested directly for minutes of compute, run the test rather than argue the assumption — the argument convinces, the check settles.
- **Every task ships an executable verification entry point.**
  A script that recomputes each promoted claim from the committed artifacts alone — hashes against the provenance record, statistics from the committed tables, the summary's evidence table checked against the artifacts mechanically — printing claim / recomputed / pass-fail, runnable on a laptop in about a minute; and a notebook, committed **without outputs**, whose cells recompute the same claims **inline and self-contained** (standard-library hashing, direct table reads, explicit arithmetic — nothing imported from the project's own code), so the reviewer reads exactly what is computed and watches it pass.
  A notebook that only calls the script relocates the trust instead of discharging it; the script is the notebook's final cross-check cell and the continuous-integration form (a test runs the same checks), not its body.
  Trust comes from re-derivation the reader performs, not from narrative the writer asserts; a number recomputed at read time cannot drift the way a number transcribed across documents can.
- **State an assumption with its quantitative exposure, never as a bare hedge.**
  Say what value would have to obtain for the conclusion to change and whether that value is plausible — "the draws are not independent" alone tells a reader nothing; "the dependence would need to inflate the null variance 3,700-fold to lose significance" tells them everything.

## 4. Git and collaboration

- **One task, one branch — and one at a time.**
  Every task gets its own branch, cut from the trunk (`project-docs` until it merges; `main` after that) and named for the task slug, so the branch, the task folder, and the spec-tree entry share one identity.
  Process changes a task motivates are made on the task's branch and land with it, not slipped onto the trunk.
  Finish the work, open the pull request, merge it, then branch again: branching before the current piece lands puts two half-finished versions of the same documents on disk, and every question after that — which number is current, which file is real — has more than one answer.
- **The branch carries every script in its result's provenance chain.**
  Data downloads, input builds, model builds: if a script produced anything the task's result depends on, it is on the task's branch before the result is promoted.
  A result whose producing scripts live elsewhere — an old worktree, an uncommitted file on the cluster — cannot be regenerated from the branch that claims it.
- **Commit messages state the root cause, not just the change**, written so a future reader can tell what happened and why without re-deriving it from the diff.
  Do not put result numbers in a commit message — reference the output and the document that reports it, so there is one place a number can be corrected.
- **Push to `origin`** (the `lagillenwater` fork).
  **Never `upstream`** (`greenelab`) without being explicitly told.
  Greene Lab standard beyond this fork: fork-per-contributor, pull requests only, minimum one lab-member approval, one functional area per pull request.

## 5. Documentation discipline

The rules these serve are in [`docs/SPEC.md`](SPEC.md); this is what they require in practice:

- A new task = a `docs/tasks/<task-slug>/design.md` **and** a spec-tree branch in `SPEC.md` pointing at it, in the same change.
- Reversing an architecture, method, or data source are updated in the task-specific `design.md` and `plan.md` docs, with the old choices and reasons for changing dated and appended to the bottom of the documents. No exceptions.
- A change to what the project asks, how work is done, or where a rung stands = `README.md` updated in the same change.
  The README is the only document most readers open, so it is the one that goes stale first and the one that misleads most when it does. Update the summary, not just the document it summarises.
- A number change = `docs/STATE.md` updated in the same change that produced it, and that entry carries its three links: **spec** (the task's `design.md`), **code** (the commits or files that changed), **outputs** (the promoted artifacts + their provenance record).
  A status claim with no link to the spec that motivated it, the code that produced it, and the artifact it came from is a claim a future session has to re-derive from git log. 
- **Landed documents reference only landed work.**
  A rung above the current one is referenced in the sense of `docs/SPEC.md` — what the ladder will ask — never through unlanded code or documents a reviewer of this repository cannot open.
  Work from the archived development lineage is cited only by its archive branch or archive file, labeled as archive.
  A forward reference a reviewer cannot follow reads as missing work, and a worktree reference reads as work that exists when, for the repository, it does not.
  Files on the cluster are cited by repository-relative path plus "on Alpine" (scratch locations in `$USER` form) — an absolute site path embeds a username and a mount layout no reviewer can use.
- **Every acronym is spelled out at its first use, in every report document.**
  A task's design, verification, review, audit, and summary documents are read by people who did not write them; an undefined acronym is a claim the reader cannot check.
  Common file-format and tooling abbreviations are not exempt.

## 6. Definition of done

- **A bug fix**: a real test (imports the actual function) passes locally; if it touches a promoted number, that result is re-run, re-promoted with a fresh provenance record, and the document that reported the old number gets a correction banner — never a silent edit.
- **A new capability**: went through brainstorm → spec → plan in one `docs/tasks/<task-slug>/` folder, indexed in `SPEC.md`, with `STATE.md`'s entry linking spec, code, and outputs.
- **A claim of "complete"**: backed by command output you actually ran, not by what the code should do.
- **Any task**: the README reflects it. A rung that landed, a rule that arrived, a status that moved — if a reader would learn it from the three documents, the README says it too.
- **A task heading to its pull request**: the drift audit passed after verification and **before promotion** — design read clause by clause against the landed tree, every departure aligned or recorded through a fix wave, and a re-audit by a fresh reader confirming it, all in `audit.md` — and the plain-language summary exists (`summary.md`: hypothesis, evidence with its table, controls and figure, conclusions, scripts touched). The pull request opens as a draft after both, and is marked ready only after a final human review of the task documents.

## 7. Failure modes this process exists to prevent

Written down because each has already cost this project real time, and because the lesson outlives the specific confusion.

- **A newer, more authoritative-looking document can itself be the source of drift.**
  Recency is not accuracy: a new spec must be checked against the code and the specs it supersedes, not just written down.
- **Renaming a concept without a forward pointer leaves every prior document silently ambiguous.**
  A code-level rename with a working alias reads as complete while every document using the old name keeps meaning something slightly different.
  A rename is a two-line fix in the documents that used the old name; do it at rename time, not retroactively.
- **A parallel implementation that bypasses an abstraction silently drops that abstraction's guarantees.**
  When a driver is written to call a library directly because the registry path is inconvenient, every guarantee the registry provided — leakage filtering, shared partitions — quietly stops applying, and no output says so.
  If a guarantee matters, it belongs where it cannot be routed around.
- **A reversal with no decision entry is invisible to everyone but git log.**
  Reverting an architecture, a method or a data source and writing nothing down leaves the next reader to rediscover it from commit archaeology, or to reinstate what was deliberately removed.

## Changes

This manual's dated change lineage lives in [`docs/decisions.md`](decisions.md) (moved 2026-08-31).
