"""latent-trainer-serve: single-replay hosting pipeline for the SC2 Latent Trainer.

Wraps :mod:`latent_trainer.inference` (extraction -> features -> encode ->
path -> feedback) behind a CLI, and later a FastAPI app, so an end user can
run the trained model on their own .SC2Replay file.
"""
