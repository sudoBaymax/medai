import torch
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix
import monai
from training.scripts.dataset import MedicalSegDataset
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def calculate_dice(pred, target):
    """Calculate Dice coefficient."""
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calculate_iou(pred, target):
    """Calculate Intersection over Union."""
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def calculate_sensitivity_specificity(pred, target):
    """Calculate sensitivity and specificity."""
    pred = (pred > 0.5).float()
    tn, fp, fn, tp = confusion_matrix(target.cpu().numpy().ravel(), 
                                     pred.cpu().numpy().ravel()).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

def evaluate_model(model, dataloader, device):
    """Evaluate model on the given dataloader."""
    model.eval()
    metrics = {
        'dice': [],
        'iou': [],
        'sensitivity': [],
        'specificity': []
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            
            # Calculate metrics for each sample in batch
            for i in range(outputs.size(0)):
                pred = outputs[i, 0]
                target = masks[i, 0]
                
                metrics['dice'].append(calculate_dice(pred, target).item())
                metrics['iou'].append(calculate_iou(pred, target).item())
                sens, spec = calculate_sensitivity_specificity(pred, target)
                metrics['sensitivity'].append(sens)
                metrics['specificity'].append(spec)
    
    # Calculate mean metrics
    return {k: np.mean(v) for k, v in metrics.items()}

def main():
    # Load configuration
    config = load_config('training/configs/base_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset and dataloader
    test_dataset = MedicalSegDataset(config['data']['test_path'], is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Load model
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['num_classes'],
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    
    # Load best model weights
    checkpoint = torch.load(Path(config['logging']['save_dir']) / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Path(config['logging']['save_dir']) / 'evaluation_metrics.csv', index=False)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 