# Methodology and official-template update (2026-09-05)

This report supersedes stale status statements in the historical Phase 1 report.

- Official Elsevier elsarticle 3.4 was downloaded via the publisher's LaTeX page;
  the unmodified class and numeric style are included. SEGAN's guide returned 403.
- Portable TinyTeX now compiles the actual LaTeX manuscript and standalone
  supplementary appendices. Tectonic's package hosts did not resolve; no
  ReportLab conversion is used for these new deliverables.
- Kevin David Ortega, Michael Cifuentes Molano and Daniel Zapata Yarce are
  included in the requested order. No affiliations, emails, CRediT assignments,
  funding or conflict statements have been invented.
- Added circuit connectivity, control/data architecture, control activation and
  electrical response, critical-day voltage heatmap and controlled outage plots.
- Generated bus, line, load, DER, event and operating-result tables from files.
- Five appendices specify units, benchmark parameters, controller equations,
  event definitions and reproduction/pending-experiment details.
- S0-S4: 5 cases x 100 days x 96 intervals = 48,000 AC solves from one synthetic
  generator seed. These are not independent statistical replications.
- E0-E2: 3 matched cases x 96 intervals = 288 AC solves, all convergent. E1
  disables batteries during 18:00-19:00; E2 opens R1-R2 and disconnects 17 buses,
  producing 40.846933 kWh unserved demand. These are quasi-static outage tests,
  not short-circuit or protection simulations. Both replay runs reproduced the
  same numerical results; repeated execution is not counted as a new experiment.
- Corrected sub-meter conversion to `60 * sum(Wh per minute) / 1000` kW and
  regenerated processed data from the unchanged original file/checksum.
  Residual columns are not used by forecasting features or power-flow multipliers.
- Recorded the perfect-synthetic-PV assumption in S3/S4 and the distinction
  between the gradient-boosting evaluation and the persistence-based controller.

Outstanding work: network-aware NSGA-II, comparator optimizer, 30 seeds, physical
degradation, S5/S6, sensitivity/ablation and statistical inference. The draft is
not submission-ready merely because the template now compiles.

Commands: `python scripts/build_methodology_assets.py`,
`python scripts/run_contingency_pilot.py`, `python -m pytest -q`,
`python scripts/check_manuscript.py`, `scripts/build_paper.ps1`,
`python scripts/package_paper.py`. PDF visual/contact-sheet and text checks are
recorded in `results/pdf_quality_audit.json`.
