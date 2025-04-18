import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
import yaml
import time
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(model_name, gpu_id):
    """Train a specific model on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create model-specific config
    config = load_config('training/configs/base_config.yaml')
    config['logging']['tensorboard_dir'] = f"tensorboard_logs/{model_name}"
    config['logging']['save_dir'] = f"results/{model_name}"
    
    # Create directories
    Path(config['logging']['tensorboard_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging']['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # Start training
    print(f"Starting training for {model_name} on GPU {gpu_id}")
    os.system(f"python training/scripts/train_{model_name}.py --config training/configs/base_config.yaml")

def main():
    # List of models to train
    models = [
        "unet",
        "unetpp",
        "deeplabv3",
        "densenet",
        "resunet",
        "maskrcnn",
        "pspnet",
        "fcn",
        "segformer",
        "attentionunet",
        "sam",
        "transunet"
    ]
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    # Set up multiprocessing
    mp.set_start_method("spawn")
    processes = []
    
    # Launch training processes
    for idx, model in enumerate(models):
        gpu_id = idx % num_gpus
        p = mp.Process(target=train_model, args=(model, gpu_id))
        p.start()
        processes.append(p)
        
        # Add a small delay between launches to prevent resource contention
        time.sleep(5)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All training processes completed!")

if __name__ == "__main__":
    main() 