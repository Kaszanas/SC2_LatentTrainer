"""SC2 Latent Trainer — run with ``python -m latent_trainer``.

Delegates to the Click CLI defined in :mod:`latent_trainer.train`.
All command-line arguments are handled by Click — run
``python -m latent_trainer --help`` for usage information.
"""

from latent_trainer.train import main

if __name__ == "__main__":
    main()
