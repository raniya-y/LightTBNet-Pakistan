import torch
import torch.nn as nn
from pathlib import Path

# ==================== MODEL ARCHITECTURE ====================
class CustomResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding='same')
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[1], kernel_size=1, stride=1, padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels[1])
        )
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.relu = nn.ReLU()

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = self.bn1(self.relu(self.conv1(input)))
        input = self.bn2(self.relu(self.conv2(input)))
        input = torch.cat([input, shortcut], dim=1)
        return input


class LightTBNet_4blocks(nn.Module):
    def __init__(self, in_channels, outputs):
        super().__init__()
       
        layer0 = nn.Sequential(
            CustomResBlock(in_channels, [16, 16]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer1 = nn.Sequential(
            CustomResBlock(32, [32, 32]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer2 = nn.Sequential(
            CustomResBlock(64, [64, 64]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer3 = nn.Sequential(
            CustomResBlock(128, [128, 128]),
            nn.MaxPool2d(kernel_size=2, stride=None, padding=0),
        )
        layer4 = nn.Sequential(
            nn.Conv2d(256, out_channels=16, kernel_size=1, padding='valid'),
            nn.ReLU(),
        )

        self.features = nn.Sequential()
        self.features.add_module('layer0', layer0)
        self.features.add_module('layer1', layer1)
        self.features.add_module('layer2', layer2)
        self.features.add_module('layer3', layer3)
        self.features.add_module('layer4', layer4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
        )

    def forward(self, input):
        input = self.features(input)
        input = self.classifier(input)
        return input


# ==================== SAVE WEIGHTS ====================
def save_model_weights(model, output_path="./models/model_best.pth"):
    """
    Save only the model weights (not optimizer state)
    """
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)
   
    # Save only state_dict (weights)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, output_path)
   
    print(f"✅ Model weights saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")


# ==================== MAIN ====================
def main():
    print("="*60)
    print("💾 SAVING MODEL WEIGHTS")
    print("="*60)
   
    # --- START MODIFIED SECTION ---
    
    # Define the base directory where all your results are stored
    # Based on your train.py output:
    BASE_DIR = Path("./results/exp_21112025_212342/LightTBNet_4blocks/Adam/FocalLoss") 
    
    # 1. Look for all 'model_best.pth' files across all folds
    print("\n📁 Searching for 'model_best.pth' checkpoints across all CV folds...")
    
    # Use glob to find all 'fold_X' directories containing 'model_best.pth'
    found_paths = list(BASE_DIR.glob('fold_*/model_best.pth'))
    
    if not found_paths:
        print(f"❌ No 'model_best.pth' found in any subfolder under: {BASE_DIR}")
        print("\n💡 Please ensure BASE_DIR path and experiment name are correct.")
        return
        
    print(f"✓ Found {len(found_paths)} best model checkpoints.")
    print("-" * 30)

    # 2. Present options and let the user select a fold
    print("   Select the fold checkpoint you want to save (weights only):")
    for i, path in enumerate(found_paths):
        # path.parent.name will be 'fold_0', 'fold_1', etc.
        print(f"   [{i + 1}] {path.parent.name} ({path})")
    print("-" * 30)
    
    try:
        selection = input("Enter the number of the fold to use (e.g., 1 for fold_0): ")
        index = int(selection) - 1
        if 0 <= index < len(found_paths):
            checkpoint_path = found_paths[index]
            print(f"Selected fold: {checkpoint_path.parent.name}")
        else:
            print("❌ Invalid selection. Exiting.")
            return
    except ValueError:
        print("❌ Invalid input. Exiting.")
        return
        
    # --- END MODIFIED SECTION ---
   
    print(f"\n📂 Loading checkpoint from: {checkpoint_path}")
   
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print("   ✓ Checkpoint loaded")
   
    # Initialize model
    print("\n🧠 Initializing model architecture...")
    model = LightTBNet_4blocks(in_channels=1, outputs=2)
    print("   ✓ Model created")
   
    # Load weights into model (rest of the logic remains the same)
    print("\n🔄 Loading weights into model...")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✓ Weights loaded from checkpoint")
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("   ✓ Weights loaded from checkpoint")
    else:
        print("   ⚠️  No model_state_dict found, assuming checkpoint IS the state_dict")
        model.load_state_dict(checkpoint)
        print("   ✓ Weights loaded")
   
    # Save weights only
    print("\n💾 Saving weights only...")
    output_path = "./models/model_best.pth"
    save_model_weights(model, output_path)
   
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"\n📁 Weights saved to: {output_path}")
    print(f"\n🚀 You can now use this with the Streamlit app!")
    print(f"\nUpdate your Streamlit app with:")
    print(f'   model_path = "{output_path}"')


if __name__ == "__main__":
    main() 