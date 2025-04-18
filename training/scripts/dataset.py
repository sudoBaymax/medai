import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MedicalSegDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Dataset class for medical image segmentation.
        
        Args:
            data_dir (str): Directory containing the images and masks
            transform (callable, optional): Optional transform to be applied on a sample
            is_train (bool): Whether this is for training or validation/testing
        """
        self.data_dir = Path(data_dir)
        self.image_paths = sorted(list(self.data_dir.glob("images/*.nii.gz")))
        self.mask_paths = sorted(list(self.data_dir.glob("masks/*.nii.gz")))
        
        assert len(self.image_paths) == len(self.mask_paths), \
            "Number of images and masks must be equal"
        
        # Default transforms if none provided
        if transform is None:
            if is_train:
                self.transform = A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.Flip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
                    A.OneOf([
                        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
                    ], p=0.3),
                    A.OneOf([
                        A.GaussNoise(p=0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        A.RandomGamma(p=0.5),
                    ], p=0.3),
                    A.Normalize(mean=0.5, std=0.5),
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=0.5, std=0.5),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = nib.load(str(self.image_paths[idx])).get_fdata()
        mask = nib.load(str(self.mask_paths[idx])).get_fdata()
        
        # Ensure 3D volumes are properly handled
        if len(image.shape) == 3:
            # Take middle slice for 2D segmentation
            middle_slice = image.shape[-1] // 2
            image = image[..., middle_slice]
            mask = mask[..., middle_slice]
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image[None, ...]
            mask = mask[None, ...]
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return {
            "image": image,
            "mask": mask,
            "image_path": str(self.image_paths[idx]),
            "mask_path": str(self.mask_paths[idx])
        } 