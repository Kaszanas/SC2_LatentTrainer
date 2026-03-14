"""Central configuration for the SC2 Latent Trainer.

Provides project-wide constants that are imported by multiple modules.

Constants
---------
LOGGING_FORMAT : str
    Standard logging format used across all training scripts.
    Includes timestamp, log level, logger name, and message.

DEFAULT_MLFLOW_URI : str
    SQLite-backed MLFlow tracking URI.  Resolved to an absolute path
    so it works correctly even when Ray workers change their working
    directory.  All modules default to this URI unless overridden via
    CLI arguments.

    To use a remote MLFlow server instead, pass ``--mlflow-uri``
    to the training CLI::

        uv run python -m latent_trainer --mlflow-uri http://localhost:5000
"""

from pathlib import Path

LOGGING_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# SQLite-backed MLFlow tracking — more robust than flat-file mlruns/
# Path.resolve() gives an absolute path so Ray workers (which change cwd)
# still write to the same database.
DEFAULT_MLFLOW_URI = f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
