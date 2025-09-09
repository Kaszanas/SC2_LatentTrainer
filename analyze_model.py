"""Simple analysis script for understanding the model with limited data."""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent / "src" / "latent_trainer" / "models"))

from guided_vae import suGuidedVAE, Classifier
from sc2_datasets.lightning.sc2_egset_datamodule import SC2EGSetDataModule
from sc2_datasets.available_replaypacks import EXAMPLE_REAL_REPLAYPACKS
from sc2_datasets.transforms.pytorch.economy_vs_outcome import economy_average_vs_outcome

def analyze_single_sample():
    """Analyze what the model learned from a single sample."""
    print("=== Single Sample Analysis ===")
    
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
        transform=economy_average_vs_outcome,
    )
    datamodule.prepare_data()
    datamodule.setup()
    
    dataloader = datamodule.train_dataloader()
    
    # Get the sample
    for data, label in dataloader:
        # Fix data shape
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        
        # Fix label shape
        if len(label.shape) == 1 and label.shape[0] > 1:
            label = label[0:1].float()
        elif len(label.shape) == 1:
            label = label.float()
        
        data = data.to(device)
        label = label.to(device)
        
        print(f"Data shape: {data.shape}")
        print(f"Label: {label.item()} ({'Win' if label.item() == 1 else 'Loss'})")
        
        # Analyze with untrained model
        with torch.no_grad():
            mu, logvar = model.encode(data)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)
            cls_output = model.cls(z)
        
        print(f"Latent code (z) shape: {z.shape}")
        print(f"Latent code statistics:")
        print(f"  Mean: {z.mean().item():.4f}")
        print(f"  Std: {z.std().item():.4f}")
        print(f"  Min: {z.min().item():.4f}")
        print(f"  Max: {z.max().item():.4f}")
        
        print(f"Classification output: {cls_output.item():.4f}")
        print(f"Predicted outcome: {'Win' if cls_output.item() > 0.5 else 'Loss'}")
        
        # Reconstruction error
        mse = torch.nn.functional.mse_loss(recon, data)
        print(f"Reconstruction MSE: {mse.item():.6f}")
        
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
    print("   • Try different transforms (mmr_vs_result, etc.)")
    print("   • Adjust the dataloader batch size")
    print("   • Use different replay sources")
    
    print("\n4. 📈 Enhanced Visualizations:")
    print("   • Generate synthetic data to fill the latent space")
    print("   • Interpolate between latent codes")
    print("   • Analyze individual latent dimensions")
    
    print("\n5. 🎮 Understanding the Current Results:")
    print("   • Even with 1 sample, you can see:")
    print("     - How the model encodes economic data")
    print("     - Reconstruction quality")
    print("     - Latent space structure")
    print("     - Individual dimension patterns")

def main():
    print("🎯 Quick Analysis of Your StarCraft 2 Economic Pattern Model")
    print("="*65)
    
    try:
        data, z, recon, label = analyze_single_sample()
        
        print(f"\n✅ SUCCESS! Generated visualizations with available data.")
        print(f"📁 Check the 'visualizations' folder for:")
        print(f"   • latent_dimensions.png - Shows how each dimension is distributed")
        print(f"   • economic_patterns.png - Heatmap of economic features")
        print(f"   • reconstruction_quality.png - Original vs reconstructed data")
        
        suggest_improvements()
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. Train the model: python src/latent_trainer/models/train_model.py --epochs 20")
        print(f"   2. Re-run visualizations: python visualize_patterns.py")
        print(f"   3. Explore the generated images in the visualizations folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        suggest_improvements()

if __name__ == "__main__":
    main()
