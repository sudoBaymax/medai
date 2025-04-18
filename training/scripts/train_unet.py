import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import monai
from tqdm import tqdm
import numpy as np
from dataset import MedicalSegDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(config):
    return monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['num_classes'],
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

def get_loss_fn():
    return monai.losses.DiceCELoss(to_onehot_y=True, sigmoid=True)

def get_optimizer(model, config):
    return optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

def get_scheduler(optimizer, config):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0
    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()
    
    return val_loss / len(dataloader)

def main():
    # Load configuration
    config = load_config('training/configs/base_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets and dataloaders
    train_dataset = MedicalSegDataset(config['data']['train_path'], is_train=True)
    val_dataset = MedicalSegDataset(config['data']['val_path'], is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model, loss function, optimizer, and scheduler
    model = get_model(config).to(device)
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Initialize TensorBoard
    writer = SummaryWriter(config['logging']['tensorboard_dir'])
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, Path(config['logging']['save_dir']) / 'best_model.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    writer.close()

if __name__ == '__main__':
    main() 