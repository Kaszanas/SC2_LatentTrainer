"""
Exploratory data analysis of the events cache: paper-ready figures and tables.

Run ``python -m latent_trainer.analysis.eda``. Every output is computed into a
table first (saved as Parquet) and plotted from that table, so ``--replot``
redraws everything after a style change without recomputing. Tables meant for
the paper are also written as ``booktabs`` LaTeX. All style lives in
:mod:`latent_trainer.analysis.eda.style`.

Scope rule: outputs that could inform a modelling choice use the training
split only; only the descriptive outputs (composition, label balance, game
length, split comparability, player overlap) look at val/test.
"""
