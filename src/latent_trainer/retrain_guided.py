"""Standalone entry point for retraining the Guided VAE with fixed best-known hyperparameters.

Run as:
    python src/latent_trainer/retrain_guided.py --experiment_name my_run --help
"""

from latent_trainer.models.train_guided import main

if __name__ == "__main__":
    main()
