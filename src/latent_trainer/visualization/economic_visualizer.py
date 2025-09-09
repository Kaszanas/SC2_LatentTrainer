"""Advanced visualization module for Guided VAE economic patterns."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..models.guided_vae import suGuidedVAE, Classifier
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS, EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.pytorch.economy_vs_outcome import economy_average_vs_outcome


class EconomicPatternVisualizer:
    """Visualizer for economic patterns learned by the Guided VAE."""
    
    def __init__(self, model_path=None, n_vae_dis=16):
        """Initialize the visualizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = suGuidedVAE(n_vae_dis=n_vae_dis).to(self.device)
        self.classifier = Classifier(n_vae_dis=n_vae_dis).to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        # Set up data module
        self.datamodule = SC2EGSetDataModule(
            unpack_dir="./data/unpack",
            download_dir="./data/download",
            download=True,
            replaypacks=SC2EGSET_DATASET_REPLAYPACKS,
            transform=economy_average_vs_outcome,
        )
        self.datamodule.prepare_data()
        self.datamodule.setup()
        
        # Setup plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def load_model(self, model_path):
        """Load a trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
        self.model.eval()
        self.classifier.eval()
    
    def extract_latent_representations(self, num_samples=500):
        """Extract latent representations from the dataset."""
        self.model.eval()
        latent_codes = []
        labels = []
        reconstructions = []
        original_data = []
        
        dataloader = self.datamodule.train_dataloader()
        sample_count = 0
        
        with torch.no_grad():
            for i, (data, label) in enumerate(dataloader):
                if sample_count >= num_samples:
                    break
                    
                # Fix data shape
                if len(data.shape) == 3:
                    data = data.unsqueeze(0)
                
                # Fix label shape
                if len(label.shape) == 1 and label.shape[0] > 1:
                    # Multiple labels - take the first one
                    label = label[0:1].float()
                elif len(label.shape) == 1:
                    label = label.float()
                
                data = data.to(self.device)
                label = label.to(self.device)
                
                # Get latent representation
                mu, logvar = self.model.encode(data)
                z = self.model.reparameterize(mu, logvar)
                recon = self.model.decode(z)
                
                latent_codes.append(z.cpu().numpy())
                labels.append(label.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())
                original_data.append(data.cpu().numpy())
                
                sample_count += 1
                
                if sample_count % 50 == 0:
                    print(f"Processed {sample_count} samples...")
        
        print(f"Total samples collected: {sample_count}")
        
        if sample_count == 0:
            raise ValueError("No samples collected from dataloader!")
        
        return (np.vstack(latent_codes), 
                np.hstack(labels), 
                np.vstack(reconstructions),
                np.vstack(original_data))
    
    def plot_latent_space_2d(self, latent_codes, labels, method='tsne', save_path=None):
        """Visualize latent space in 2D using t-SNE or PCA."""
        n_samples = latent_codes.shape[0]
        print(f"Plotting latent space with {n_samples} samples using {method.upper()}")
        
        if n_samples < 2:
            print(f"Warning: Only {n_samples} samples available. Skipping {method.upper()} plot.")
            return None
        
        if method == 'tsne':
            # Adjust perplexity based on number of samples
            perplexity = min(30, max(5, n_samples // 4))
            print(f"Using perplexity: {perplexity}")
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedding = reducer.fit_transform(latent_codes)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latent_codes)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=labels, cmap='RdYlBu', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Game Outcome (0=Loss, 1=Win)')
        plt.title(f'Latent Space Visualization ({method.upper()}) - {n_samples} samples')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return embedding
    
    def plot_latent_dimensions(self, latent_codes, labels, save_path=None):
        """Plot distribution of each latent dimension colored by outcome."""
        n_dims = latent_codes.shape[1]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(n_dims):
            ax = axes[i]
            
            # Separate by outcome
            wins = latent_codes[labels == 1, i]
            losses = latent_codes[labels == 0, i]
            
            ax.hist(losses, alpha=0.7, label='Losses', bins=30, color='red')
            ax.hist(wins, alpha=0.7, label='Wins', bins=30, color='blue')
            
            ax.set_title(f'Latent Dimension {i+1}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_dims, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_economic_patterns_heatmap(self, original_data, labels, save_path=None):
        """Plot heatmap of average economic patterns by outcome."""
        # Check if we have both wins and losses
        unique_labels = np.unique(labels)
        n_wins = np.sum(labels == 1)
        n_losses = np.sum(labels == 0)
        
        print(f"Samples: {n_wins} wins, {n_losses} losses")
        
        if n_wins == 0 and n_losses == 0:
            print("No data available for heatmap")
            return
        
        # Reshape data from (batch, 66, 2, 39) to (batch, 66*2*39) for easier averaging
        # Then reshape back to (66, 2*39) for visualization
        reshaped_data = original_data.reshape(original_data.shape[0], 66, -1)  # (batch, 66, 78)
        
        fig, axes = plt.subplots(1, min(3, len(unique_labels) + 1), figsize=(18, 6))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        plot_idx = 0
        
        if n_wins > 0:
            wins_data = reshaped_data[labels == 1].mean(axis=0)  # (66, 78)
            im1 = axes[plot_idx].imshow(wins_data, aspect='auto', cmap='Blues')
            axes[plot_idx].set_title(f'Average Economic Patterns - Wins ({n_wins} samples)')
            axes[plot_idx].set_xlabel('Time Steps × Players')
            axes[plot_idx].set_ylabel('Economic Features')
            plt.colorbar(im1, ax=axes[plot_idx])
            plot_idx += 1
        
        if n_losses > 0:
            losses_data = reshaped_data[labels == 0].mean(axis=0)  # (66, 78)
            im2 = axes[plot_idx].imshow(losses_data, aspect='auto', cmap='Reds')
            axes[plot_idx].set_title(f'Average Economic Patterns - Losses ({n_losses} samples)')
            axes[plot_idx].set_xlabel('Time Steps × Players')
            axes[plot_idx].set_ylabel('Economic Features')
            plt.colorbar(im2, ax=axes[plot_idx])
            plot_idx += 1
        
        # Only show difference if we have both wins and losses
        if n_wins > 0 and n_losses > 0:
            wins_data = reshaped_data[labels == 1].mean(axis=0)
            losses_data = reshaped_data[labels == 0].mean(axis=0)
            difference = wins_data - losses_data
            
            im3 = axes[plot_idx].imshow(difference, aspect='auto', cmap='RdBu_r', 
                               vmin=-np.abs(difference).max(), vmax=np.abs(difference).max())
            axes[plot_idx].set_title('Difference (Wins - Losses)')
            axes[plot_idx].set_xlabel('Time Steps × Players')
            axes[plot_idx].set_ylabel('Economic Features')
            plt.colorbar(im3, ax=axes[plot_idx])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_reconstruction_quality(self, original_data, reconstructions, labels, n_samples=5, save_path=None):
        """Plot comparison between original and reconstructed data."""
        # Select random samples
        actual_samples = min(n_samples, len(original_data))
        indices = np.random.choice(len(original_data), actual_samples, replace=False)
        
        fig, axes = plt.subplots(actual_samples, 3, figsize=(15, 3*actual_samples))
        
        # Handle single sample case
        if actual_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Reshape from (66, 2, 39) to (66, 78) for visualization
            original = original_data[idx].reshape(66, -1)
            reconstructed = reconstructions[idx].reshape(66, -1)
            difference = original - reconstructed
            outcome = "Win" if labels[idx] == 1 else "Loss"
            
            # Original
            im1 = axes[i, 0].imshow(original, aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'Original - {outcome}')
            if i == 0:
                axes[i, 0].set_ylabel('Economic Features')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Reconstructed
            im2 = axes[i, 1].imshow(reconstructed, aspect='auto', cmap='viridis')
            axes[i, 1].set_title(f'Reconstructed - {outcome}')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Difference
            max_diff = np.abs(difference).max()
            im3 = axes[i, 2].imshow(difference, aspect='auto', cmap='RdBu_r', 
                                  vmin=-max_diff, vmax=max_diff)
            axes[i, 2].set_title(f'Difference - {outcome}')
            plt.colorbar(im3, ax=axes[i, 2])
            
            if i == actual_samples - 1:
                for j in range(3):
                    axes[i, j].set_xlabel('Time Steps × Players')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_latent_explorer(self, latent_codes, labels, save_path=None):
        """Create an interactive 3D plot of the latent space."""
        # Use first 3 dimensions for 3D plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=latent_codes[:, 0],
                y=latent_codes[:, 1],
                z=latent_codes[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=labels,
                    colorscale='RdYlBu',
                    showscale=True,
                    colorbar=dict(title="Game Outcome")
                ),
                text=[f'Outcome: {"Win" if l == 1 else "Loss"}' for l in labels],
                hovertemplate='Dim 1: %{x:.2f}<br>Dim 2: %{y:.2f}<br>Dim 3: %{z:.2f}<br>%{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Interactive 3D Latent Space Explorer',
            scene=dict(
                xaxis_title='Latent Dimension 1',
                yaxis_title='Latent Dimension 2',
                zaxis_title='Latent Dimension 3'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        return fig
    
    def generate_comprehensive_report(self, output_dir="visualizations", num_samples=50):
        """Generate a comprehensive visualization report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Extracting latent representations...")
        latent_codes, labels, reconstructions, original_data = self.extract_latent_representations(num_samples)
        
        n_samples = len(labels)
        print(f"Successfully extracted {n_samples} samples")
        
        if n_samples == 0:
            print("❌ No samples extracted! Check your data pipeline.")
            return None
        
        print("Creating latent space visualizations...")
        # 2D visualizations - only if we have enough samples
        if n_samples >= 10:
            self.plot_latent_space_2d(latent_codes, labels, method='pca', 
                                    save_path=output_path / 'latent_space_pca.png')
            
            if n_samples >= 30:  # Only do t-SNE if we have enough samples
                self.plot_latent_space_2d(latent_codes, labels, method='tsne', 
                                        save_path=output_path / 'latent_space_tsne.png')
            else:
                print(f"Skipping t-SNE: need at least 30 samples, got {n_samples}")
        else:
            print(f"Skipping 2D plots: need at least 10 samples, got {n_samples}")
        
        print("Plotting latent dimensions...")
        self.plot_latent_dimensions(latent_codes, labels,
                                  save_path=output_path / 'latent_dimensions.png')
        
        print("Creating economic pattern heatmaps...")
        self.plot_economic_patterns_heatmap(original_data, labels,
                                          save_path=output_path / 'economic_patterns.png')
        
        print("Analyzing reconstruction quality...")
        n_recon_samples = min(5, n_samples)
        self.plot_reconstruction_quality(original_data, reconstructions, labels, 
                                       n_samples=n_recon_samples,
                                       save_path=output_path / 'reconstruction_quality.png')
        
        if n_samples >= 3:  # Need at least 3 samples for 3D plot
            print("Creating interactive 3D explorer...")
            self.create_interactive_latent_explorer(latent_codes, labels,
                                                   save_path=output_path / 'interactive_latent_explorer.html')
        else:
            print(f"Skipping 3D plot: need at least 3 samples, got {n_samples}")
        
        print(f"All visualizations saved to {output_path}")
        
        return {
            'latent_codes': latent_codes,
            'labels': labels,
            'reconstructions': reconstructions,
            'original_data': original_data
        }


if __name__ == "__main__":
    # Create visualizer
    visualizer = EconomicPatternVisualizer()
    
    # Generate comprehensive report
    data = visualizer.generate_comprehensive_report()
    
    print("Visualization complete! Check the 'visualizations' folder for all plots.")
