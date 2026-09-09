"""Single-replay inference pipeline: extract -> features -> encode -> path -> feedback.

Lets an end user run the trained model on their own ``.SC2Replay`` file,
without needing the bulk cached dataset used for training. See
:func:`latent_trainer.inference.pipeline.predict_replay` for the entry point.
"""
