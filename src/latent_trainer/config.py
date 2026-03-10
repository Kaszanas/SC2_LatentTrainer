"""Central configuration for the SC2 Latent Trainer.

Provides project-wide constants such as the logging format and
default MLFlow tracking URI.
"""

from pathlib import Path

LOGGING_FORMAT = (
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# SQLite-backed MLFlow tracking — more robust than flat-file mlruns/
DEFAULT_MLFLOW_URI = f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
