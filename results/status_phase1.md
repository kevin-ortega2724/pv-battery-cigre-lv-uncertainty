# Phase 1 status report — 2026-09-04

## Initial inventory

The workspace initially contained only an empty `.git` directory: no commit, README, dataset, source code, tests, results or manuscript. Git required a command-local `safe.directory` override because the sandbox user differs from the directory owner; global Git configuration was not changed.

## Work completed

- Created a 41-file reproducible project scaffold, local `.venv`, pinned requirements, Conda environment and YAML configurations.
- Implemented raw UCI parsing, chronological ordering, missing-value handling, quality reporting and the 14-of-15 aggregation rule.
- Corrected the residual-load unit interpretation: sub-meter channels are Wh per minute (numerically average W over a minute); after aggregation, residual kW is `Global_active_power - sum(submeters)/1000`, and 15-min energy is residual kW times 0.25 h.
- Added six unit/integration tests.
- Implemented the pandapower CIGRE LV benchmark and exported all bus, line, transformer, load, switch and external-grid parameters to CSV.
- Compared Newton--Raphson and backward/forward sweep on identical benchmark data.
- Created an Elsevier `elsarticle` manuscript with background, explicit TODO markers and 16 selected references.
- Created a literature matrix with DOI-verification status and stated limitations.

## Commands executed

```powershell
.\.venv\Scripts\python.exe -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib pyyaml pytest pandapower
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_cigre_base.py
.\.venv\Scripts\python.exe scripts\check_manuscript.py
.\.venv\Scripts\python.exe scripts\run_data_audit.py
git -c safe.directory='C:/Users/USUARIO/Documents/ChatGPT/SISTEMAS DE POTENCIA' status --short
```

## Verified results

| Check | Result |
|---|---:|
| Tests | 16 passed |
| Raw rows | 2,075,259 |
| Rows with missing numeric measurements | 25,979 |
| Duplicate timestamps | 0 |
| Continuous numeric-missing runs | 71 |
| Largest numeric outage | 7,226 min = 5.0181 days |
| Valid 15-min intervals (at least 14 observations) | 136,593 |
| 15-min forecast MAE / RMSE | 0.2562 / 0.4642 kW |
| 15-min nominal-80% PICP | 80.19% |
| 1-h forecast MAE / PICP | 0.4160 kW / 78.95% |
| 6-h forecast MAE / PICP | 0.4885 kW / 78.02% |
| 24-h forecast MAE / PICP | 0.4940 kW / 79.08% |
| 1-h rolling-origin 2009 MAE / RMSE / PICP | 0.4316 / 0.7036 kW / 80.44% |
| CIGRE buses / lines / transformers / loads | 44 / 37 / 3 / 15 |
| NR minimum voltage | 0.9122689603 p.u. |
| NR active line losses | 21.8217536 kW |
| NR maximum line loading | 31.7162179% |
| NR maximum transformer loading | 85.2546404% |
| NR grid import | 714.9292201 kW |
| Maximum NR–BFSW voltage difference | 2.5936e-9 p.u. |
| NR–BFSW line-loss difference | 8.3851e-7 kW |
| S0 temporal AC flows converged | 9,600 / 9,600 |
| S0 minimum voltage / undervoltage intervals | 0.8953 p.u. / 911 (9.49%) |
| S0 maximum line / transformer loading | 35.48% / 100.87% |
| S0 peak / mean grid import | 742.15 / 181.05 kW |
| S0 line-loss energy | 5,078.62 kWh (1.169% of import) |
| OpenDSS maximum / mean voltage difference | 4.61e-6 / 2.26e-6 p.u. |
| pandapower / OpenDSS total base losses | 28.3292 / 28.3272 kW |
| Independent relative loss difference | 0.0071% |
| S1 loss / undervoltage / peak reduction vs S0 | 12.96% / 4.39% / 0.00% |
| S2 loss / undervoltage / peak reduction vs S0 | 21.23% / 9.88% / 2.05% |
| S2 SOC range / simultaneous charge-discharge | 10–90% / 0 intervals |
| S3 loss / undervoltage / peak reduction vs S0 | 11.49% / 7.14% / 6.12% |
| S4 loss / undervoltage / peak reduction vs S0 | 8.92% / 4.17% / 1.93% |
| S3/S4 AC convergence and SOC checks | 9,600/9,600 each; 10–90%; no simultaneous charge/discharge |

## Unverified reported pilot figures

The raw-data counts and 136,593 valid intervals have now been reproduced from a 132,960,755-byte source with SHA-256 `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7`. The reported forecast figures are approximately reproduced only for a 15-min horizon, not for the requested 1-h horizon. The 45.92 kW peak, 0.945 p.u. voltage, 68.88 A current and 17.72 kWh daily losses remain unverified because the original pilot feeder implementation was not present in the initially empty repository.

## Problems and limitations

1. Raw data remain external to Git; collaborators must obtain the UCI file independently and verify its recorded SHA-256.
2. No TeX engine is installed. Static checks confirm the official class declaration, included files, balanced braces and all 16 citation keys, but this is not a PDF compilation.
3. Independent OpenDSS validation is balanced and positive-sequence; unbalanced four-wire validation remains outside this phase.
4. The unmodified CIGRE case has a 0.9123 p.u. minimum voltage, below the planned 0.95 p.u. operating bound. Benchmark assumptions and loading must be documented before any rescaling.
5. The initial literature set is a defensible seed, not the final target of 40–60 sources. Two records legitimately have no DOI (JMLR article and CIGRE brochure).
6. `numba` is intentionally not required; power flows run with `numba=False` for a smaller reproducible environment.

## Gate decision

NSGA-II was not implemented. The raw-data audit and independent CIGRE power-flow gates are satisfied. S0--S4 are complete. S3 is the strongest tested peak-reduction policy, while S2 remains strongest for line losses and undervoltage exposure. The q90-envelope S4 pilot does not dominate S3 and must not be presented as the final stochastic optimizer.

## Elsevier/Overleaf transfer

The manuscript has been transferred to the official `elsarticle` preprint structure and its abstract, results, discussion and conclusions now contain the verified S0--S4 findings. `results/SEGAN_Overleaf_submission.zip` contains the self-contained manuscript source, bibliography and figures. Only author-controlled editorial metadata remains marked `TODO`. Static validation passes; local PDF compilation is unavailable because this host has no TeX engine.

## Recommended next action

Before launching NSGA-II, freeze the S0--S4 assumptions and define the multi-objective decision space, constraint handling, common scenario set and computational budget. The final study should add network-aware placement/dispatch and quantify Pareto uncertainty rather than optimizing only aggregate net demand.
