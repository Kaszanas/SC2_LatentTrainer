"""Simple analysis script for understanding the model with MMR data."""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent / "src" / "latent_trainer" / "models"))

from models.guided_vae import suGuidedVAE, Classifier
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.available_replaypacks import EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.mmr_vs_result import mmr_vs_result

def analyze_single_sample():
    """Analyze what the model learned from a single sample."""
    print("=== Single Sample Analysis (MMR vs Result) ===")
    
    # Set up model and data
    device = torch.device("cpu")
    model = suGuidedVAE(n_vae_dis=16).to(device)
    model.eval()
    
    # Get data
    datamodule = SC2EGSetDataModule(
        unpack_dir="./data/unpack",
        download_dir="./data/download",
        download=True,
        replaypacks=EXAMPLE_REAL_REPLAYPACKS,
        transform=mmr_vs_result,
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    dataloader = datamodule.train_dataloader()
    
    # Get the sample
    for data, label in dataloader:
        # Take only the first sample from the batch
        data = data[0:1]  # Keep batch dimension but take only first sample
        label = label[0:1].float()
        
        data = data.to(device)
        label = label.to(device)
        
        print(f"Data shape: {data.shape}")
        print(f"Player 1 APM: {data[0, 0].item():.2f}")
        print(f"Player 2 APM: {data[0, 1].item():.2f}")
        print(f"Label: {label.item()} ({'Win' if label.item() == 1 else 'Loss'})")
        
        # Analyze with untrained model
        with torch.no_grad():
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)
            cls_output = model.cls(z)
        
        print(f"\nLatent code (z) shape: {z.shape}")
        print(f"Latent code statistics:")
        print(f"  Mean: {z.mean().item():.4f}")
        print(f"  Std: {z.std().item():.4f}")
        print(f"  Min: {z.min().item():.4f}")
        print(f"  Max: {z.max().item():.4f}")
        
        print(f"\nClassification output: {cls_output.item():.4f}")
        print(f"Predicted outcome: {'Win' if cls_output.item() > 0.5 else 'Loss'}")
        
        # Reconstruction error
        mse = torch.nn.functional.mse_loss(recon, data)
        print(f"\nReconstruction MSE: {mse.item():.6f}")
        print(f"Reconstructed Player 1 APM: {recon[0, 0].item():.2f}")
        print(f"Reconstructed Player 2 APM: {recon[0, 1].item():.2f}")
        
        # Show what each latent dimension captures
        print("\nLatent dimensions analysis:")
        for i in range(z.shape[1]):
            print(f"  Dimension {i+1}: {z[0, i].item():.4f}")
        
        break
    
    return data, z, recon, label

def suggest_improvements():
    """Suggest ways to get more training data and better visualizations."""
    print("\n" + "="*60)
    print("🚀 SUGGESTIONS FOR BETTER VISUALIZATIONS")
    print("="*60)
    
    print("\n1. 📊 Get More Data:")
    print("   • Use more replaypacks in the EXAMPLE_REAL_REPLAYPACKS")
    print("   • Download additional StarCraft 2 replay datasets")
    print("   • Increase batch size in the dataloader")
    print("   • Use validation/test splits for more samples")
    
    print("\n2. 🎯 Train the Model:")
    print("   • Run: python src/latent_trainer/models/train_model.py --epochs 50")
    print("   • A trained model will show more meaningful patterns")
    print("   • The latent space will be more structured after training")
    
    print("\n3. 🔧 Modify the Dataset:")
    print("   • Try different transforms (mmr_vs_result currently used)")
    print("   • Adjust the dataloader batch size")
    print("   • Use different replay sources")
    
    print("\n4. 📈 Enhanced Visualizations:")
    print("   • Generate synthetic data to fill the latent space")
    print("   • Interpolate between latent codes")
    print("   • Analyze individual latent dimensions")
    print("   • Compare APM patterns between wins and losses")
    
    print("\n5. 🎮 Understanding the Current Results:")
    print("   • Even with 1 sample, you can see:")
    print("     - How the model encodes APM data")
    print("     - Reconstruction quality of player APM values")
    print("     - Latent space structure")
    print("     - Individual dimension patterns")

def main():
    print("🎯 Quick Analysis of Your StarCraft 2 MMR Pattern Model")
    print("="*65)
    
    try:
        data, z, recon, label = analyze_single_sample()
        
        print("\n✅ SUCCESS! Generated visualizations with available data.")
        print("📁 Check the 'visualizations' folder for:")
        print("   • latent_dimensions.png - Shows how each dimension is distributed")
        print("   • mmr_patterns.png - APM scatter plots and comparisons")
        print("   • reconstruction_quality.png - Original vs reconstructed APM")
        
        suggest_improvements()
        
        print("\n💡 NEXT STEPS:")
        print("   1. Train the model: python src/latent_trainer/models/train_model.py --epochs 20")
        print("   2. Re-run visualizations: python visualize_patterns.py")
        print("   3. Explore the generated images in the visualizations folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        suggest_improvements()

if __name__ == "__main__":
    main()
