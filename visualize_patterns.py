"""Script to run economic pattern visualizations."""

import sys
import os
from pathlib import Path

# Add the models directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent / "src" / "latent_trainer" / "models"))
sys.path.append(str(Path(__file__).parent / "src" / "latent_trainer" / "visualization"))

from economic_visualizer import EconomicPatternVisualizer

def main():
    print("=== StarCraft 2 MMR Pattern Visualizer ===")
    print()
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Check if trained model exists
    model_path = Path("output/best_model.pth")
    if model_path.exists():
        print(f"Found trained model at {model_path}")
        visualizer = EconomicPatternVisualizer(model_path=str(model_path), transform='mmr')
    else:
        print("No trained model found. Using untrained model for demonstration.")
        print("Train a model first using: python src/latent_trainer/models/train_model.py --epochs 10")
        visualizer = EconomicPatternVisualizer(transform='mmr')
    
    print()
    print("Generating visualizations...")
    print("This may take a few minutes...")
    
    try:
        # Generate comprehensive report
        data = visualizer.generate_comprehensive_report(
            output_dir=str(output_dir),
            num_samples=100  # Reduced for faster processing and to avoid sample issues
        )
        
        print()
        print("=" * 50)
        print("🎉 Visualization Complete!")
        print("=" * 50)
        print()
        print("Generated visualizations:")
        print("📊 latent_space_tsne.png - 2D t-SNE projection of latent space")
        print("📊 latent_space_pca.png - 2D PCA projection of latent space") 
        print("📊 latent_dimensions.png - Distribution of each latent dimension")
        print("📊 mmr_patterns.png - APM patterns by outcome (Win/Loss)")
        print("📊 reconstruction_quality.png - Original vs reconstructed APM comparison")
        print("🌐 interactive_latent_explorer.html - Interactive 3D latent space explorer")
        print()
        print(f"All files saved in: {output_dir.absolute()}")
        print()
        print("🔍 Analysis Summary:")
        print(f"   • Processed {len(data['labels'])} game samples")
        print(f"   • Win rate: {data['labels'].mean():.1%}")
        print(f"   • Latent space dimensions: {data['latent_codes'].shape[1]}")
        print()
        print("💡 Next steps:")
        print("   • Open the interactive HTML file in your browser")
        print("   • Examine the latent dimension distributions")
        print("   • Look for patterns in the APM scatter plots")
        print("   • Compare original vs reconstructed APM values")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        print("Make sure you have the required data and dependencies installed.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
