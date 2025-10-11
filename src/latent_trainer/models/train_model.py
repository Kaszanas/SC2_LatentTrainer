"""Train models module."""

from __future__ import print_function
import click
import logging
from pathlib import Path
import os
from tqdm import tqdm

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

from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS, EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result
from sc2_datasets.transforms.pytorch.economy_vs_outcome import (
    economy_average_vs_outcome,
)

from sc2_datasets.transforms.utils import select_outcome_1v1


def train_supervised(epoch, model, model_c, optimizer, optimizer_c, dataloader, w_cls, device):
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
    
    print(
        "====> Epoch: {} reconstruction loss: {:.4f} Cls loss: {:.4f} acc: {:.2f} cls1 loss: {:.4f} cls2 loss: {:.4f} acc1: {:.2f} acc2: {:.2f} valid samples: {}".format(
            epoch,
            re_loss / total_valid_samples,
            cls_error,
            100.0 * correct / total_valid_samples,
            cls1_error,
            cls2_error,
            100.0 * correct1 / total_valid_samples,
            100.0 * correct2 / total_valid_samples,
            total_valid_samples
        )
    )
    
    # Return total loss for model saving
    total_loss = re_loss / total_valid_samples + cls_error + cls1_error + cls2_error
    return total_loss


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

def main(batch_size, output, epochs, nz, cls, num_workers, test_interval, lr, weight_decay, lr_c, weight_decay_c, transform):
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

    model = suGuidedVAE(n_vae_dis=nz).to(device)
    model_c = Classifier(n_vae_dis=nz).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    optimizer_c = optim.Adam(
        model_c.parameters(), lr=lr_c, weight_decay=weight_decay_c
    )
    
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