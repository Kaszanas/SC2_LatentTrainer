"""Train models module."""

from __future__ import print_function
import click
import optuna
import logging
from pathlib import Path
import os

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from latent_trainer.models.guided_vae import Classifier, suGuidedVAE
from latent_trainer.models.losses import loss_supervised

from latent_trainer.config import LOGGING_FORMAT

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule

from sc2_datasets.available_replaypacks import EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

from sc2_datasets.transforms.utils import select_outcome_1v1
import lightning as L
from lightning.pytorch import Trainer


class LitGuidedVAE(L.LightningModule):
    def __init__(self, n_vae_dis=16, lr=1e-4, weight_decay=1e-5, lr_c=1e-4, weight_decay_c=1e-4, w_cls=200.0):
        super().__init__()
        self.save_hyperparameters()
        
        # Set manual optimization flag to use multiple optimizers
        self.automatic_optimization = False
        
        # Main VAE model
        self.model = suGuidedVAE(n_vae_dis=n_vae_dis)
        
        # Classifier for adversarial training
        self.classifier = Classifier(n_vae_dis=n_vae_dis)
        
        # Store hyperparameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_c = lr_c
        self.weight_decay_c = weight_decay_c
        self.w_cls = w_cls
        
        # Track metrics
        self.total_valid_samples = 0
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Get optimizers for manual optimization
        opt_vae, opt_cls, opt_adv = self.optimizers()
        
        data, label = batch
        
        # Process labels
        if label.dtype == torch.int8:
            label = label.float()
        
        if len(label.shape) == 1:
            label = label.unsqueeze(1)
            
        # Filter out invalid labels (-1)
        valid_indices = (label != -1).squeeze()
        
        if valid_indices.sum() == 0:
            self.log('train_skip_batch', 1)
            return None
            
        # Extract valid data
        valid_data = data[valid_indices]
        valid_label = label[valid_indices]
        
        # Ensure labels are in valid range [0,1]
        valid_label = torch.clamp(valid_label, 0, 1)
        
        # Track valid samples
        batch_valid_samples = valid_indices.sum().item()
        self.total_valid_samples += batch_valid_samples
        
        # Step 1: VAE step
        # Clear gradients for optimizer 1
        opt_vae.zero_grad()
        
        recon_batch, mu, logvar, re = self.model(valid_data)
        loss_list = loss_supervised(recon_batch, valid_data, mu, logvar)
        vae_loss = loss_list[0]
        loss_cls = F.binary_cross_entropy(re, valid_label, reduction="sum")
        vae_total_loss = vae_loss + loss_cls * self.w_cls
        
        # Calculate accuracy
        pred = (re > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.size(0) * 100
        
        # Log metrics with on_step=True and on_epoch=True for better TensorBoard tracking
        self.log('train_vae_loss', vae_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_cls_loss', loss_cls, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_vae_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_valid_samples', batch_valid_samples, on_epoch=True)
        
        # Manually backpropagate and optimize
        self.manual_backward(vae_total_loss)
        opt_vae.step()
        
        # Step 2: Classifier step
        # Clear gradients for optimizer 2
        opt_cls.zero_grad()
        
        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar).detach()
        z = z[:, 1:]
        cls1 = self.classifier(z)
        cls_loss = F.binary_cross_entropy(cls1, valid_label, reduction="sum")
        cls_loss *= self.w_cls
        
        # Calculate accuracy
        pred = (cls1 > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.size(0) * 100
        
        # Log metrics with on_step=True and on_epoch=True for better TensorBoard tracking
        self.log('train_c_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_c_acc', acc, prog_bar=True, on_step=True, on_epoch=True)
        
        # Manually backpropagate and optimize
        self.manual_backward(cls_loss)
        opt_cls.step()
        
        # Step 3: Adversarial step
        # Clear gradients for optimizer 3
        opt_adv.zero_grad()
        
        mu, logvar = self.model.encode(valid_data)
        z = self.model.reparameterize(mu, logvar)
        z = z[:, 1:]
        cls2 = self.classifier(z)
        label1 = torch.empty_like(valid_label).fill_(0.5)
        adv_loss = F.binary_cross_entropy(cls2, label1, reduction="sum")
        adv_loss *= self.w_cls
        
        # Log metrics with on_step=True and on_epoch=True for better TensorBoard tracking
        self.log('train_adv_loss', adv_loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Manually backpropagate and optimize
        self.manual_backward(adv_loss)
        opt_adv.step()
        
        # Return combined loss for logging purposes only
        return vae_total_loss + cls_loss + adv_loss

    def on_train_epoch_end(self):
        self.log('epoch_valid_samples', self.total_valid_samples)
        self.total_valid_samples = 0  # Reset for next epoch
    
    def validation_step(self, batch, batch_idx):
        data, label = batch
        
        # Process labels (similar to training_step)
        if label.dtype == torch.int8:
            label = label.float()
        
        if len(label.shape) == 1:
            label = label.unsqueeze(1)
            
        # Filter out invalid labels (-1)
        valid_indices = (label != -1).squeeze()
        
        if valid_indices.sum() == 0:
            return None
            
        # Extract valid data
        valid_data = data[valid_indices]
        valid_label = label[valid_indices]
        
        # Ensure labels are in valid range [0,1]
        valid_label = torch.clamp(valid_label, 0, 1)
        
        # Run forward pass
        recon_batch, mu, logvar, re = self.model(valid_data)
        
        # Calculate losses
        loss_list = loss_supervised(recon_batch, valid_data, mu, logvar)
        vae_loss = loss_list[0]
        loss_cls = F.binary_cross_entropy(re, valid_label, reduction="sum")
        
        # Calculate accuracy
        pred = (re > 0.5).float()
        acc = pred.eq(valid_label).sum().item() / valid_label.size(0) * 100
        
        # Log metrics with enhanced settings for better visualization
        self.log('val_vae_loss', vae_loss, prog_bar=True, sync_dist=True)
        self.log('val_cls_loss', loss_cls, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        
        return vae_loss

    def configure_optimizers(self):
        # Optimizer for the VAE
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Optimizer for the Classifier
        optimizer_c = optim.Adam(
            self.classifier.parameters(),
            lr=self.lr_c,
            weight_decay=self.weight_decay_c
        )
        
        return [optimizer, optimizer_c], []  # No schedulers

# Training is now handled in the LitGuidedVAE class training_step method

# The objective function is now defined inside the main function when using Optuna

# CLICK command line interface
@click.command()
@click.option('--batch-size', '-b', default=128, help='input batch size for training (default: 128)', type=int)
@click.option('--output', default='output', help='output directory for results')
@click.option('--epochs', default=10, help='number of epochs to train (default: 10)', type=int)
@click.option('--nz', default=16, help='bottleneck size', type=int)
@click.option('--cls', default=200.0, help='classification error weight for supervised Guided-VAE', type=float)
@click.option('--num_workers', default=0, help='number of workers for dataloader', type=int)
@click.option('--test_interval', default=1, help='interval for testing', type=int)
@click.option('--lr', default=1e-4, help='learning rate', type=float)
@click.option('--weight_decay', default=1e-5, help='weight decay', type=float)
@click.option('--lr_c', default=1e-4, help='classifier learning rate(in supervised version)', type=float)
@click.option('--weight_decay_c', default=1e-4, help='classifier weight decay(in supervised version)', type=float)
@click.option('--transform', default='mmr_vs_result', type=click.Choice(['mmr_vs_result', 'economy_average_vs_outcome', 'average_player_stats', 'select_outcome_1v1']), help='which transform to use')
@click.option('--optuna', 'use_optuna', is_flag=True, help='use Optuna for hyperparameter optimization')
@click.option('--n_trials', default=20, help='number of Optuna trials (only used with --optuna)', type=int)
@click.option('--optuna_epochs', default=3, help='number of epochs per trial for Optuna (only used with --optuna)', type=int)
@click.option('--optuna_db', default='sqlite:///optuna_study.db', help='Optuna database URL for dashboard (only used with --optuna)', type=str)
@click.option('--study_name', default='vae_optimization', help='Optuna study name (only used with --optuna)', type=str)

def main(batch_size, output, epochs, nz, cls, num_workers, test_interval, lr, weight_decay, lr_c, weight_decay_c, transform, use_optuna, n_trials, optuna_epochs, optuna_db, study_name):
    """Main function to parse arguments and start training."""
    # Set up more verbose logging to help diagnose issues
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT)
    logging.info("Starting model training...")

    download_path = Path("./data/download").resolve().as_posix()
    unpack_path = Path("./data/unpack").resolve().as_posix()

    logging.info(f"{download_path=}")
    logging.info(f"{unpack_path=}")

    if not os.path.exists(output):
        os.mkdir(output)

    torch.manual_seed(1024)
    
    # Select the appropriate transform
    transform_map = {
        'mmr_vs_result': mmr_vs_result,
        'economy_average_vs_outcome': economy_average_vs_outcome,
        'select_outcome_1v1': select_outcome_1v1,
        # 'average_player_stats': average_player_stats,  # Add this when available
    }
    selected_transform = transform_map.get(transform, mmr_vs_result)
    
    # Set up lighting datamodule
    sc2_egset_datamodule = SC2EGSetDataModule(
        unpack_dir="./data/unpack",
        download_dir="./data/download",
        download=True,
        replaypacks=EXAMPLE_REAL_REPLAYPACKS,
        transform=selected_transform,
        batch_size=batch_size,
        num_workers=num_workers
    )
    sc2_egset_datamodule.prepare_data()
    sc2_egset_datamodule.setup()
    
    # Set up callbacks
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=os.path.join(output, 'checkpoints'),
        filename='model-{epoch:02d}-{val_vae_loss:.4f}',
        monitor='val_vae_loss',
        mode='min',
        save_last=True,
        save_top_k=3,
    )
    
    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor='val_vae_loss', 
        patience=5,
        mode='min'
    )
    
    # TensorBoard logger
    tb_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir=output,
        name='tensorboard_logs',
        version=None  # Use root directory directly for cleaner access
    )
    
    # Check if Optuna optimization is requested
    if use_optuna:
        logging.info("Starting Optuna hyperparameter optimization...")
        logging.info(f"Running {n_trials} trials with {optuna_epochs} epochs each")
        logging.info(f"Study will be saved to: {optuna_db}")
        logging.info(f"Study name: {study_name}")
        logging.info(f"To view Optuna Dashboard, run: optuna-dashboard {optuna_db}")
        
        def objective(trial):
            # Suggest hyperparameters
            nz_trial = trial.suggest_int('nz', 8, 64)
            cls_trial = trial.suggest_float('cls', 0.1, 10.0)
            lr_trial = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
            weight_decay_trial = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            lr_c_trial = trial.suggest_float('lr_c', 1e-5, 1e-3, log=True)
            weight_decay_c_trial = trial.suggest_float('weight_decay_c', 1e-6, 1e-3, log=True)
            
            # Create Lightning model with suggested hyperparameters
            model = LitGuidedVAE(
                n_vae_dis=nz_trial,
                lr=lr_trial,
                weight_decay=weight_decay_trial,
                lr_c=lr_c_trial,
                weight_decay_c=weight_decay_c_trial,
                w_cls=cls_trial
            )
            
            # Pruning callback for Optuna
            pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor='val_vae_loss'
            )
            
            # Logger for this trial
            trial_logger = L.pytorch.loggers.TensorBoardLogger(
                save_dir=os.path.join(output, 'tensorboard_logs', 'optuna_trials'),
                name=f'trial_{trial.number}',
                version=None  # Use root directory directly for cleaner access
            )
            
            # Create trainer with fewer epochs for trial
            trainer = Trainer(
                max_epochs=optuna_epochs,
                logger=trial_logger,
                enable_progress_bar=True,
                callbacks=[pruning_callback],
                accelerator='auto',
                devices=1,
                enable_checkpointing=False,
                log_every_n_steps=10
            )
            
            # Get the dataloaders directly to avoid the prepare_data issue
            train_loader = sc2_egset_datamodule.train_dataloader()
            val_loader = sc2_egset_datamodule.val_dataloader()
            
            # Train model with explicit dataloaders instead of the datamodule
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            
            # Return the final validation loss
            return trainer.callback_metrics['val_vae_loss'].item()
        
        # Create Optuna study with database storage
        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_db,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True  # Allow resuming if study exists
        )
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Print results and log hyperparameters to TensorBoard
        hparam_dict = study.best_trial.params
        metric_dict = {"val_vae_loss": study.best_trial.value}
        tb_logger.log_hyperparams(hparam_dict, metric_dict)
        
        logging.info("Optuna optimization completed!")
        logging.info(f"Best trial: {study.best_trial.number}")
        logging.info(f"Best value (loss): {study.best_trial.value}")
        logging.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logging.info(f"  {key}: {value}")
        
        logging.info(f"\n{'='*60}")
        logging.info("To view the Optuna Dashboard with all trials, run:")
        logging.info(f"  optuna-dashboard {optuna_db}")
        logging.info(f"{'='*60}\n")
        
        # Save best hyperparameters
        best_params_path = os.path.join(output, 'best_hyperparameters.txt')
        with open(best_params_path, 'w') as f:
            f.write(f"Optuna Study: {study_name}\n")
            f.write(f"Database: {optuna_db}\n")
            f.write(f"Total trials: {len(study.trials)}\n")
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best value (loss): {study.best_trial.value}\n")
            f.write("\nBest hyperparameters:\n")
            for key, value in study.best_trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nTo view dashboard: optuna-dashboard {optuna_db}\n")
        logging.info(f"Best hyperparameters saved to {best_params_path}")
        
        # Train final model with best hyperparameters
        logging.info("Training final model with best hyperparameters...")
        best_params = study.best_trial.params
        
        # Create model with best parameters
        model = LitGuidedVAE(
            n_vae_dis=best_params['nz'],
            lr=best_params['lr'],
            weight_decay=best_params['weight_decay'],
            lr_c=best_params['lr_c'],
            weight_decay_c=best_params['weight_decay_c'],
            w_cls=best_params['cls']
        )
        
        # Train with full epochs
        final_trainer = Trainer(
            max_epochs=epochs,
            logger=tb_logger,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping],
            accelerator='auto',
            devices=1,
            log_every_n_steps=10
        )
        
        # Get the dataloaders directly
        train_loader = sc2_egset_datamodule.train_dataloader()
        val_loader = sc2_egset_datamodule.val_dataloader()
        
        # Train with explicit dataloaders
        final_trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Save final model in PyTorch format for compatibility
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.model.state_dict(),
            'classifier_state_dict': model.classifier.state_dict(),
            'hyperparameters': best_params,
        }, f'{output}/final_model.pth')
        
        print(f"Training completed with Optuna! Final model saved to {output}/final_model.pth")
        print(f"Best checkpoints saved in: {os.path.join(output, 'checkpoints')}")
        
    else:
        # Normal training without Optuna
        logging.info("Starting normal training (without Optuna)...")
        logging.info(f"TensorBoard logs will be saved to: {os.path.join(output, 'tensorboard_logs')}")
        logging.info(f"To view logs, run: tensorboard --logdir={os.path.join(output, 'tensorboard_logs')}")
        
        # Create model
        model = LitGuidedVAE(
            n_vae_dis=nz,
            lr=lr,
            weight_decay=weight_decay,
            lr_c=lr_c,
            weight_decay_c=weight_decay_c,
            w_cls=cls
        )
        
        # Create trainer
        trainer = Trainer(
            max_epochs=epochs,
            logger=tb_logger,
            enable_progress_bar=True,
            callbacks=[checkpoint_callback, early_stopping],
            accelerator='auto',
            devices=1,
            check_val_every_n_epoch=test_interval,
            log_every_n_steps=10
        )
        
        # Get the dataloaders directly to avoid the prepare_data issue
        train_loader = sc2_egset_datamodule.train_dataloader()
        val_loader = sc2_egset_datamodule.val_dataloader()
        
        # Train the model with explicit dataloaders instead of the datamodule
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Save final model in PyTorch format for compatibility
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.model.state_dict(),
            'classifier_state_dict': model.classifier.state_dict(),
            'loss': trainer.callback_metrics.get('train_vae_loss', float('inf')).item(),
        }, f'{output}/final_model.pth')
        
        print(f"Training completed! Final model saved to {output}/final_model.pth")
        print(f"Best checkpoints saved in: {os.path.join(output, 'checkpoints')}")



if __name__ == "__main__":
    # The Click decorator will parse command line arguments and pass them to main
    main()