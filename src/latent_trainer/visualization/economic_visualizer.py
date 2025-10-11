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

from models.guided_vae import suGuidedVAE, Classifier

from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.available_replaypacks import SC2EGSET_DATASET_REPLAYPACKS, EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.pytorch.economy_vs_outcome import economy_average_vs_outcome
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result


class EconomicPatternVisualizer:
    """Visualizer for MMR patterns learned by the Guided VAE."""
    
    def __init__(self, model_path=None, n_vae_dis=16, transform='mmr'):
        """Initialize the visualizer.
        
        Args:
            model_path: Path to trained model checkpoint
            n_vae_dis: Number of VAE latent dimensions
            transform: Type of transform to use ('mmr' or 'economy')
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = suGuidedVAE(n_vae_dis=n_vae_dis).to(self.device)
        self.classifier = Classifier(n_vae_dis=n_vae_dis).to(self.device)
        self.transform_type = transform
        
        if model_path:
            self.load_model(model_path)
        
        # Choose transform based on type
        if transform == 'mmr':
            transform_fn = mmr_vs_result
        else:
            transform_fn = economy_average_vs_outcome
        
        # Set up data module
        self.datamodule = SC2EGSetDataModule(
            unpack_dir="./data/unpack",
            download_dir="./data/download",
            download=True,
            replaypacks=EXAMPLE_REAL_REPLAYPACKS,
            transform=transform_fn,
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
                
                # Skip if either data or label is None (filtered by transform)
                if data is None or label is None:
                    continue
                    
                # Handle batched data - process each sample individually
                batch_size = data.shape[0] if len(data.shape) > 1 else 1
                
                for j in range(batch_size):
                    if sample_count >= num_samples:
                        break
                    
                    # Extract single sample
                    if batch_size > 1:
                        sample_data = data[j:j+1]
                        sample_label = label[j:j+1] if len(label.shape) > 0 else label.unsqueeze(0)
                    else:
                        sample_data = data.unsqueeze(0) if len(data.shape) == 1 else data
                        sample_label = label.unsqueeze(0) if len(label.shape) == 0 else label
                    
                    sample_data = sample_data.to(self.device)
                    sample_label = sample_label.to(self.device).float()
                    
                    # Get latent representation
                    mu, logvar = self.model.encode(sample_data)
                    z = self.model.reparameterize(mu, logvar)
                    recon = self.model.decode(z)
                    
                    latent_codes.append(z.cpu().numpy().squeeze())
                    labels.append(sample_label.cpu().numpy().squeeze())
                    reconstructions.append(recon.cpu().numpy().squeeze())
                    original_data.append(sample_data.cpu().numpy().squeeze())
                    
                    sample_count += 1
                    
                    if sample_count % 50 == 0:
                        print(f"Processed {sample_count} samples...")
        
        print(f"Total samples collected: {sample_count}")
        
        if sample_count == 0:
            raise ValueError("No samples collected from dataloader!")
        
        # Stack arrays appropriately based on their shapes
        latent_codes_arr = np.array(latent_codes)
        labels_arr = np.array(labels)
        reconstructions_arr = np.array(reconstructions)
        original_data_arr = np.array(original_data)
        
        return (latent_codes_arr, 
                labels_arr, 
                reconstructions_arr,
                original_data_arr)
    
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
        """Plot heatmap of average patterns by outcome (works for both MMR and economic data)."""
        # Check if we have both wins and losses
        unique_labels = np.unique(labels)
        n_wins = np.sum(labels == 1)
        n_losses = np.sum(labels == 0)
        
        print(f"Samples: {n_wins} wins, {n_losses} losses")
        
        if n_wins == 0 and n_losses == 0:
            print("No data available for heatmap")
            return
        
        # Check data shape to determine if it's MMR or economic data
        if len(original_data.shape) == 2 and original_data.shape[1] == 2:
            # MMR data: shape (batch, 2) - just 2 APM values
            self._plot_mmr_patterns(original_data, labels, n_wins, n_losses, save_path)
        else:
            # Economic data: shape (batch, 66, 2, 39)
            self._plot_economic_patterns(original_data, labels, n_wins, n_losses, save_path)
    
    def _plot_mmr_patterns(self, original_data, labels, n_wins, n_losses, save_path):
        """Plot MMR/APM patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Scatter plot of Player 1 APM vs Player 2 APM
        if n_wins > 0:
            wins_data = original_data[labels == 1]
            axes[0].scatter(wins_data[:, 0], wins_data[:, 1], 
                          alpha=0.6, c='blue', label=f'Wins ({n_wins})', s=50)
        
        if n_losses > 0:
            losses_data = original_data[labels == 0]
            axes[0].scatter(losses_data[:, 0], losses_data[:, 1], 
                          alpha=0.6, c='red', label=f'Losses ({n_losses})', s=50)
        
        axes[0].set_xlabel('Player 1 APM', fontsize=12)
        axes[0].set_ylabel('Player 2 APM', fontsize=12)
        axes[0].set_title('APM Distribution by Outcome', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Bar plot of average APM by outcome
        categories = []
        p1_means = []
        p2_means = []
        colors = []
        
        if n_wins > 0:
            wins_data = original_data[labels == 1]
            categories.append('Wins')
            p1_means.append(wins_data[:, 0].mean())
            p2_means.append(wins_data[:, 1].mean())
            colors.append('blue')
        
        if n_losses > 0:
            losses_data = original_data[labels == 0]
            categories.append('Losses')
            p1_means.append(losses_data[:, 0].mean())
            p2_means.append(losses_data[:, 1].mean())
            colors.append('red')
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1].bar(x - width/2, p1_means, width, label='Player 1 APM', alpha=0.8)
        axes[1].bar(x + width/2, p2_means, width, label='Player 2 APM', alpha=0.8)
        
        axes[1].set_ylabel('Average APM', fontsize=12)
        axes[1].set_title('Average APM by Outcome', fontsize=14)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_economic_patterns(self, original_data, labels, n_wins, n_losses, save_path):
        """Plot economic patterns for economy data."""
        # Reshape data from (batch, 66, 2, 39) to (batch, 66*2*39) for easier averaging
        # Then reshape back to (66, 2*39) for visualization
        reshaped_data = original_data.reshape(original_data.shape[0], 66, -1)  # (batch, 66, 78)
        
        fig, axes = plt.subplots(1, min(3, len(np.unique(labels)) + 1), figsize=(18, 6))
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
        
        # Check data shape to determine visualization type
        if len(original_data.shape) == 2 and original_data.shape[1] == 2:
            # MMR data: shape (batch, 2)
            self._plot_mmr_reconstruction(original_data, reconstructions, labels, indices, save_path)
        else:
            # Economic data: shape (batch, 66, 2, 39)
            self._plot_economic_reconstruction(original_data, reconstructions, labels, indices, save_path)
    
    def _plot_mmr_reconstruction(self, original_data, reconstructions, labels, indices, save_path):
        """Plot reconstruction quality for MMR data."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Original vs Reconstructed scatter
        ax = axes[0, 0]
        for idx in indices:
            outcome = "Win" if labels[idx] == 1 else "Loss"
            color = 'blue' if labels[idx] == 1 else 'red'
            ax.scatter(original_data[idx, 0], original_data[idx, 1], 
                      marker='o', s=100, alpha=0.7, c=color, label=f'Original ({outcome})')
            ax.scatter(reconstructions[idx, 0], reconstructions[idx, 1], 
                      marker='x', s=100, alpha=0.7, c=color, label=f'Reconstructed')
            
            # Draw line connecting original to reconstruction
            ax.plot([original_data[idx, 0], reconstructions[idx, 0]], 
                   [original_data[idx, 1], reconstructions[idx, 1]], 
                   'k--', alpha=0.3)
        
        ax.set_xlabel('Player 1 APM', fontsize=12)
        ax.set_ylabel('Player 2 APM', fontsize=12)
        ax.set_title('Original vs Reconstructed APM', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction error per sample
        ax = axes[0, 1]
        errors = np.sqrt(((original_data[indices] - reconstructions[indices])**2).sum(axis=1))
        colors = ['blue' if labels[idx] == 1 else 'red' for idx in indices]
        bars = ax.bar(range(len(indices)), errors, color=colors, alpha=0.7)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Reconstruction Error (L2)', fontsize=12)
        ax.set_title('Reconstruction Error per Sample', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Player 1 APM comparison
        ax = axes[1, 0]
        x = range(len(indices))
        ax.plot(x, original_data[indices, 0], 'o-', label='Original', markersize=8)
        ax.plot(x, reconstructions[indices, 0], 'x--', label='Reconstructed', markersize=8)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Player 1 APM', fontsize=12)
        ax.set_title('Player 1 APM: Original vs Reconstructed', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Player 2 APM comparison
        ax = axes[1, 1]
        ax.plot(x, original_data[indices, 1], 'o-', label='Original', markersize=8)
        ax.plot(x, reconstructions[indices, 1], 'x--', label='Reconstructed', markersize=8)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Player 2 APM', fontsize=12)
        ax.set_title('Player 2 APM: Original vs Reconstructed', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_economic_reconstruction(self, original_data, reconstructions, labels, indices, save_path):
        """Plot reconstruction quality for economic data."""
        actual_samples = len(indices)
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
        
        print("Creating pattern heatmaps...")
        pattern_filename = 'mmr_patterns.png' if self.transform_type == 'mmr' else 'economic_patterns.png'
        self.plot_economic_patterns_heatmap(original_data, labels,
                                          save_path=output_path / pattern_filename)
        
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
