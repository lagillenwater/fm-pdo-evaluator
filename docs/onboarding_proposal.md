# Proposal: Automated Review Tooling for greenelab/onboarding

Status: draft, for lab discussion. Not yet proposed as a PR against
greenelab/onboarding — see the parent spec,
`docs/superpowers/specs/2026-08-22-review-workflow-automation-design.md`,
for why.

## Why

greenelab/onboarding documents a fully human code-review workflow: named
reviewers, a manual checklist, one lab-member approval. It has no
automated tooling anywhere — no CI-based linting gate visible to
reviewers, no bot review, no coverage tracking. This is being tested
first in fm-pm-evaluator (a single-author repo, so a good low-risk
testbed) before being written up here for the wider lab.

## What changes, and where

Three files in greenelab/onboarding would change:

### 1. New: `extras/automated_review_tooling.md`

```markdown
# Automated Review Tooling (Optional)

This is an optional layer on top of the review process described in the
main onboarding document — it does not change the requirement that every
PR gets at least one lab-member approval before merging.

## What it adds

- **An automated PR review bot** (e.g. CodeRabbit) comments on pull
  requests automatically when they're opened, flagging likely bugs,
  security issues, and (depending on configuration) style concerns.
  Comment-only by default — it does not block merges.
- **Coverage reporting** (e.g. Codecov) tracks what fraction of a
  repository's code is exercised by its test suite, and comments on each
  PR with how much of the *changed* code is covered. Informational by
  default — a low number is a prompt to add tests, not a blocker.
- **A pre-PR self-review step**, for repos where contributors use an AI
  coding agent: an instruction (in `CLAUDE.md`, `AGENTS.md`, or
  equivalent) to run the agent's own review capability against the diff
  before opening the PR.

These three are complementary, not redundant: the self-review step
happens before a PR exists and leaves no external trace; the bot review
happens automatically on every PR regardless of who or what wrote it,
and leaves a permanent, visible comment; coverage reporting does
something neither of the others can — track a number over time across
commits.

## Setting this up for a new repository

1. **Automated PR review.** Install the CodeRabbit GitHub App (or an
   equivalent) on your personal account (for a fork/testbed) or request
   it be installed on the `greenelab` GitHub organization (for a shared
   repository — this requires an org admin, since GitHub App
   installation is a browser-based, per-namespace action that can't be
   scripted). Commit a `.coderabbit.yaml` to the repo root rather than
   relying on dashboard-only settings, so the configuration is
   version-controlled and reviewable like any other code. Recommended
   starting defaults: `profile: chill` (flags bugs/security/logic
   issues, skips style nitpicks your linter already catches),
   `request_changes_workflow: false` (comments only, never blocks
   merge — the human-approval rule stays the actual gate).

2. **Coverage reporting.** Add a coverage-report step to your existing
   CI (e.g. `pytest --cov=<package> --cov-report=xml` for Python), then
   upload it with `codecov/codecov-action`. Commit a `codecov.yml` with
   `coverage.status.project.default.informational: true` and the same
   for `patch`, so coverage is visible without being a merge gate.
   Connect the repository at codecov.io — for a personal fork this is
   self-service; for a `greenelab`-owned repository, an org admin needs
   to install Codecov's GitHub App there too, and greenelab's org-level
   "require token for public repos" setting will likely need a
   `CODECOV_TOKEN` secret added to that specific repository (existing
   organizations default to requiring one).

3. **Pre-PR self-review.** Add a short section to the repository's
   `CLAUDE.md` (or equivalent) telling the agent to run its review
   capability before opening a PR, and what to check for (e.g. the
   five-axis lens below).

## What this costs

Both are free for **public** repositories, with caveats worth knowing
before promising anyone that it is simply free:

- **CodeRabbit** requires no application, form, or plan selection —
  installing the app on a public repo is the entire process ("receive
  free reviews forever for public repositories. No additional setup is
  required," coderabbit.ai/faq). New installs begin on a 14-day Pro+
  trial, so the plan name shown in early review comments is not what you
  settle on long-term. Free review *rate limits* scale with repository
  popularity, and public repos under 10 stars must trigger each review
  manually by commenting `@coderabbitai review` instead of getting one
  automatically when the PR opens — expect this on a new lab repo.
- **Codecov** is free for public repositories, but a `CODECOV_TOKEN`
  repository secret is likely required under greenelab's org settings
  (see "Prerequisites for org-wide adoption" below).

Ongoing maintenance is low but not zero. Configuration is
version-controlled and changes rarely; the recurring cost is judgment —
each repo's `.coderabbit.yaml` `path_instructions` should reflect what
is actually worth flagging in that codebase (e.g., do not apply
library-code scrutiny to one-off analysis scripts). Note also that an
app install only covers the namespace it was installed on, so a PR
opened against a repository whose owner has not installed the app gets
no automated review at all.
```

### 2. Addition to `extras/code_review_checklist.md`

Insert before the existing "Pride" bullet, as a short framing note:

```markdown
When reviewing a PR, it helps to check it against five axes:
**correctness** (does it do what it claims, including edge cases),
**readability** (would a lab member unfamiliar with this code
understand it), **architecture** (does it fit the rest of the
codebase, or bolt on awkwardly), **security** (credentials, injection,
unsafe deserialization, unvalidated external input), and
**performance** (obviously wasteful loops or memory use, for code that
runs at any scale). The checklist below expands on specific items
within these axes.
```

### 3. Addition to `onboarding.md`, § "Source Code, Data, and Reproducibility"

Insert after the existing "Reviewing Pull Requests" bullet:

```markdown
**Automated tooling (optional).** Repositories may additionally run
automated PR review (e.g. CodeRabbit) and coverage reporting (e.g.
Codecov) — see `extras/automated_review_tooling.md` for what these add
and how to set them up. These are additive to, not a replacement for,
the ≥1-lab-member-approval requirement above.
```

## What this does NOT change

The existing human-approval requirement (named reviewer, ≥1 lab-member
approval before merge) is untouched. Nothing here is mandatory — it's
opt-in tooling a repo maintainer can adopt, not a new onboarding rule.

## Prerequisites for org-wide adoption

GitHub Apps (both CodeRabbit and Codecov) are installed per GitHub
account or organization via a browser-based consent screen — this can't
be scripted, and a personal fork's install doesn't cover the `greenelab`
org. The install on the personal fork does not extend to `greenelab`, so
a PR opened directly against a `greenelab` repository gets no automated
review until the org installs the apps itself.

Rolling this out org-wide needs someone with app-install rights on
`greenelab` (per onboarding.md, likely Casey Greene or another org
admin) to install both apps. Default to **"Only select repositories"**
and add repos as they opt in; granting "all repositories" is an explicit
organization-level decision that should follow a look at repository
visibility and data sensitivity, since CodeRabbit requests read and
write access to code, issues, and pull requests across every repo in
scope. Budget for a `CODECOV_TOKEN` secret on any `greenelab`-owned
repository that adopts this, since Codecov's "require token for public
repos" org setting defaults to required for established organizations.

## Known adjacent gap (not addressed by this proposal)

`extras/linter_install_tutorial.md` still documents `black` + `flake8`
for local linting. fm-pm-evaluator (and other recent lab repos) have
moved to `ruff`, which replaces both. Worth a separate, focused update —
flagged here so it isn't lost, not bundled into this proposal since it's
an unrelated tooling change.

## Reference implementation

This proposal's configuration is being validated end-to-end in
fm-pm-evaluator — see that repo's `.coderabbit.yaml`, `codecov.yml`, and
`.github/workflows/ci.yml` for the working configuration, and `CLAUDE.md`
for the self-review instruction.
