"""Train models module."""

from __future__ import print_function
import click
import optuna
import logging
from pathlib import Path
import os
from tqdm import tqdm

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from latent_trainer.models.guided_vae import Classifier, suGuidedVAE
from latent_trainer.models.losses import loss_supervised

from latent_trainer.config import LOGGING_FORMAT

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule

from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS, EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

from sc2_datasets.transforms.utils import select_outcome_1v1


def train_supervised(epoch, model, model_c, optimizer, optimizer_c, dataloader, w_cls, device, writer=None):
    """Train the model for one epoch with supervision."""
    model.train()
    re_loss = 0
    cls_error = 0
    correct = 0
    cls1_error = 0
    cls2_error = 0
    correct1 = 0
    correct2 = 0
    total_valid_samples = 0
    
    for batch_idx, (data, label) in enumerate(tqdm(dataloader)):
        try:
            # Process labels
            if label.dtype == torch.int8:
                label = label.float()
            
            if len(label.shape) == 1:
                label = label.unsqueeze(1)
                
            # Filter out invalid labels (-1)
            valid_indices = (label != -1).squeeze()
            
            if valid_indices.sum() == 0:
                logging.warning(f"No valid samples after filtering at batch {batch_idx}")
                continue
                
            # Extract valid data
            valid_data = data[valid_indices]
            valid_label = label[valid_indices]
            
            # Ensure labels are in valid range [0,1]
            valid_label = torch.clamp(valid_label, 0, 1)
            
            # Track valid samples for metrics
            total_valid_samples += valid_indices.sum().item()
            
            # Move to device
            valid_data = valid_data.to(device)
            valid_label = valid_label.to(device)
            
            # Step 1: VAE training
            try:
                optimizer.zero_grad()
                recon_batch, mu, logvar, re = model(valid_data)
                loss_list = loss_supervised(recon_batch, valid_data, mu, logvar)
                loss = loss_list[0]
                loss_cls = F.binary_cross_entropy(re, valid_label, reduction="sum")
                cls_error += loss_cls
                loss += loss_cls * w_cls
                loss.backward()
                re_loss += loss_list[1].item()
                optimizer.step()
            except Exception as e:
                logging.error(f"VAE training error: {e}")
                continue
                
            # Step 2: Classifier training
            try:
                optimizer_c.zero_grad()
                z = model.reparameterize(mu, logvar).detach()
                z = z[:, 1:]
                cls1 = model_c(z)
                loss = F.binary_cross_entropy(cls1, valid_label, reduction="sum")
                cls1_error += loss.item()
                loss *= w_cls
                loss.backward()
                optimizer_c.step()
            except Exception as e:
                logging.error(f"Classifier training error: {e}")
                continue
                
            # Step 3: Adversarial training
            try:
                optimizer.zero_grad()
                mu, logvar = model.encode(valid_data)
                z = model.reparameterize(mu, logvar)
                z = z[:, 1:]
                cls2 = model_c(z)
                label1 = torch.empty_like(valid_label).fill_(0.5)
                loss = F.binary_cross_entropy(cls2, label1, reduction="sum")
                cls2_error += loss.item()
                loss *= w_cls
                loss.backward()
                optimizer.step()
            except Exception as e:
                logging.error(f"Adversarial training error: {e}")
                continue
                
            # Calculate accuracies
            try:
                pred = (re > 0.5).float()
                correct += pred.eq(valid_label).sum().item()
                
                pred = (cls1 > 0.5).float()
                correct1 += pred.eq(valid_label).sum().item()
                
                pred = (cls2 > 0.5).float()
                correct2 += pred.eq(valid_label).sum().item()
            except Exception as e:
                logging.error(f"Accuracy calculation error: {e}")
                
        except Exception as e:
            logging.error(f"General batch processing error: {e}")
            continue
    
    # Handle case with no valid samples
    if total_valid_samples == 0:
        logging.warning("No valid samples in this epoch!")
        return float('inf')
    
    # Calculate metrics
    cls_error = cls_error / total_valid_samples
    cls1_error = cls1_error / total_valid_samples
    cls2_error = cls2_error / total_valid_samples
    
    # Calculate final values for printing and logging
    re_loss_avg = re_loss / total_valid_samples
    acc = 100.0 * correct / total_valid_samples
    acc1 = 100.0 * correct1 / total_valid_samples
    acc2 = 100.0 * correct2 / total_valid_samples
    
    print(
        "====> Epoch: {} reconstruction loss: {:.4f} Cls loss: {:.4f} acc: {:.2f} cls1 loss: {:.4f} cls2 loss: {:.4f} acc1: {:.2f} acc2: {:.2f} valid samples: {}".format(
            epoch,
            re_loss_avg,
            cls_error,
            acc,
            cls1_error,
            cls2_error,
            acc1,
            acc2,
            total_valid_samples
        )
    )
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Loss/reconstruction', re_loss_avg, epoch)
        writer.add_scalar('Loss/classification', cls_error, epoch)
        writer.add_scalar('Loss/classifier1', cls1_error, epoch)
        writer.add_scalar('Loss/classifier2', cls2_error, epoch)
        writer.add_scalar('Loss/total', re_loss_avg + cls_error + cls1_error + cls2_error, epoch)
        writer.add_scalar('Accuracy/vae', acc, epoch)
        writer.add_scalar('Accuracy/classifier1', acc1, epoch)
        writer.add_scalar('Accuracy/classifier2', acc2, epoch)
        writer.add_scalar('Training/valid_samples', total_valid_samples, epoch)
    
    # Return total loss for model saving
    total_loss = re_loss_avg + cls_error + cls1_error + cls2_error
    return total_loss

def objective(trial, epochs, train_dataset, output, device):
    """Optuna objective function for hyperparameter optimization."""
    # Suggest hyperparameters
    nz = trial.suggest_int('nz', 8, 64)
    cls = trial.suggest_float('cls', 0.1, 10.0)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    lr_c = trial.suggest_float('lr_c', 1e-5, 1e-3, log=True)
    weight_decay_c = trial.suggest_float('weight_decay_c', 1e-6, 1e-3, log=True)
    
    logging.info(f"Trial {trial.number}: nz={nz}, cls={cls}, lr={lr}, weight_decay={weight_decay}, lr_c={lr_c}, weight_decay_c={weight_decay_c}")
    
    # Set up model and optimizer with suggested hyperparameters
    model = suGuidedVAE(n_vae_dis=nz).to(device)
    model_c = Classifier(n_vae_dis=nz).to(device)
    
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    optimizer_c = optim.Adam(
        model_c.parameters(), lr=lr_c, weight_decay=weight_decay_c
    )
    
    # Create TensorBoard writer for this trial
    tensorboard_dir = os.path.join(output, f'tensorboard_logs/trial_{trial.number}')
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Train for a few epochs and return the final loss
    final_loss = float('inf')
    for epoch in range(1, epochs + 1):
        epoch_loss = train_supervised(
            epoch,
            model,
            model_c,
            optimizer,
            optimizer_c,
            train_dataset,
            cls,
            device,
            writer,
        )
        final_loss = epoch_loss
        
        # Report intermediate value for pruning
        trial.report(epoch_loss, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            writer.close()
            raise optuna.exceptions.TrialPruned()
    
    writer.close()
    return final_loss


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

    device = torch.device("cpu")
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
    
    sc2_egset_datamodule = SC2EGSetDataModule(
        unpack_dir="./data/unpack",
        download_dir="./data/download",
        download=True,
        replaypacks=EXAMPLE_REAL_REPLAYPACKS,
        transform=selected_transform,
    )
    sc2_egset_datamodule.prepare_data()
    sc2_egset_datamodule.setup()
    
    # Use the dataloader directly from the datamodule
    train_dataset = sc2_egset_datamodule.train_dataloader()
    
    # Check if Optuna optimization is requested
    if use_optuna:
        logging.info("Starting Optuna hyperparameter optimization...")
        logging.info(f"Running {n_trials} trials with {optuna_epochs} epochs each")
        logging.info(f"Study will be saved to: {optuna_db}")
        logging.info(f"Study name: {study_name}")
        logging.info(f"To view Optuna Dashboard, run: optuna-dashboard {optuna_db}")
        
        # Create Optuna study with database storage
        study = optuna.create_study(
            study_name=study_name,
            storage=optuna_db,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            load_if_exists=True  # Allow resuming if study exists
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, optuna_epochs, train_dataset, output, device),
            n_trials=n_trials
        )
        
        # Print results
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
        model = suGuidedVAE(n_vae_dis=best_params['nz']).to(device)
        model_c = Classifier(n_vae_dis=best_params['nz']).to(device)
        
        optimizer = optim.Adam(
            model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay']
        )
        optimizer_c = optim.Adam(
            model_c.parameters(), lr=best_params['lr_c'], weight_decay=best_params['weight_decay_c']
        )
        
        # Create TensorBoard writer for final training
        tensorboard_dir = os.path.join(output, 'tensorboard_logs/final_model')
        writer = SummaryWriter(log_dir=tensorboard_dir)
        
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            epoch_loss = train_supervised(
                epoch,
                model,
                model_c,
                optimizer,
                optimizer_c,
                train_dataset,
                best_params['cls'],
                device,
                writer,
            )
            
            # Save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model_c.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_c_state_dict': optimizer_c.state_dict(),
                    'loss': best_loss,
                    'hyperparameters': best_params,
                }, f'{output}/best_model.pth')
                print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")
        
        writer.close()
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'classifier_state_dict': model_c.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_c_state_dict': optimizer_c.state_dict(),
            'loss': best_loss,
            'hyperparameters': best_params,
        }, f'{output}/final_model.pth')
        print(f"Training completed with Optuna! Final model saved to {output}/final_model.pth")
        
    else:
        # Normal training without Optuna
        logging.info("Starting normal training (without Optuna)...")
        
        # Create TensorBoard writer
        tensorboard_dir = os.path.join(output, 'tensorboard_logs')
        writer = SummaryWriter(log_dir=tensorboard_dir)
        logging.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
        logging.info(f"To view logs, run: tensorboard --logdir={tensorboard_dir}")
        
        model = suGuidedVAE(n_vae_dis=nz).to(device)
        model_c = Classifier(n_vae_dis=nz).to(device)
        
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        optimizer_c = optim.Adam(
            model_c.parameters(), lr=lr_c, weight_decay=weight_decay_c
        )
        
        best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            epoch_loss = train_supervised(
                epoch,
                model,
                model_c,
                optimizer,
                optimizer_c,
                train_dataset,
                cls,
                device,
                writer,
            )
            
            # Save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'classifier_state_dict': model_c.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_c_state_dict': optimizer_c.state_dict(),
                    'loss': best_loss,
                }, f'{output}/best_model.pth')
                print(f"Saved best model at epoch {epoch} with loss {best_loss:.4f}")
        
        # Close the TensorBoard writer
        writer.close()
        logging.info("TensorBoard writer closed")
        
        # Save final model
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'classifier_state_dict': model_c.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_c_state_dict': optimizer_c.state_dict(),
            'loss': best_loss,
        }, f'{output}/final_model.pth')
        print(f"Training completed! Final model saved to {output}/final_model.pth")



if __name__ == "__main__":
    main()