# AGENTS.md

Guidance for coding agents working in this repository.

## Project structure

```
DQC/
├── dqc/                  # Core package (differentiable DFT/HF)
│   ├── api/              # Public API (getxc, loadbasis, parser, properties)
│   ├── df/               # Density fitting
│   ├── grid/             # Numerical grids
│   ├── hamilton/         # Hamiltonian construction (intor, PBC)
│   ├── qccalc/           # SCF solvers (HF, KS)
│   ├── system/           # Molecular / solvated systems
│   ├── utils/            # Config, caching, periodic table, PBC helpers
│   ├── xc/               # Exchange-correlation functionals
│   ├── datasets/         # Grid quadrature data
│   ├── test/             # Package tests
│   └── benchmarks/       # Upstream timing benchmark (time_forward.py)
├── docs/                 # Sphinx documentation
├── examples/             # Usage examples
├── testing/              # Upstream integration tests
├── benchmark-pyscf/      # PySCF comparison scripts
├── pyproject.toml        # Poetry project config
└── .venv/                # Poetry virtualenv (use this)
```

**Local-only on `main` (not in `upstream/master`):** treat these separately — they are ML/diagnostics tooling around the `dqc` package, not upstream core library code.

```
├── xctrain/              # XC functional training CLI and engine
├── diagnostics/          # SCF convergence, symmetry, smearing probes
└── benchmarks/           # ML experiment scripts (pwlda_smallmol, pwlda_pblock, …)
```

## Git remotes and branches

| Remote   | URL                              | Role                          |
|----------|----------------------------------|-------------------------------|
| `origin` | `git@github.com:smvinko/DQC.git` | Personal fork (local only)    |
| `upstream` | `git@github.com:OxfordHED/dqc.git` | Canonical upstream repo   |

| Branch          | Based on          | Purpose |
|-----------------|-------------------|---------|
| `origin/main`   | fork of `upstream/master` | Substantial local changes (~36 commits ahead of upstream). Review and cherry-pick selected commits from here. |
| `origin/port-fixes` | `upstream/master` | Integration branch for porting fixes upstream. Cherry-picked changes land here before eventual push to `upstream/master`. |

**Workflow:** compare `upstream/master..main` to understand fork-only changes, then cherry-pick relevant commits onto `port-fixes` (based on `upstream/master`).

```bash
git log --oneline upstream/master..main          # commits only on main
git diff upstream/master..main -- <path>         # per-path diff
git cherry-pick <sha>                            # onto port-fixes
```

## Environment

Use the existing Poetry environment in `.venv`. Do not create a new venv.

```bash
source .venv/bin/activate
# or prefix commands:
.venv/bin/python -m pytest dqc/test/
```

Install/sync only if dependencies are missing: `poetry install`

## Remote push policy

**Never push to any remote unless the user explicitly asks.**

- Do not `git push origin …`
- Do not `git push upstream …`

All work stays local until the user says otherwise. Fetching from remotes is fine; pushing is not.
