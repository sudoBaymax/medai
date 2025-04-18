from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import SimpleITK as sitk
import pydicom
import nibabel as nib
import numpy as np
from pathlib import Path

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def preprocess_dicom(input_path, output_path):
    """Preprocess DICOM files and convert to NIfTI format."""
    try:
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(input_path))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Basic preprocessing
        image = sitk.Normalize(image)
        
        # Save as NIfTI
        sitk.WriteImage(image, str(output_path))
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def preprocess_nifti(input_path, output_path):
    """Preprocess NIfTI files."""
    try:
        # Read NIfTI
        nii_img = nib.load(str(input_path))
        img_data = nii_img.get_fdata()
        
        # Basic preprocessing
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # Save processed image
        processed_nii = nib.Nifti1Image(img_data, nii_img.affine)
        nib.save(processed_nii, str(output_path))
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def process_files(**context):
    """Process all medical image files in the input directory."""
    input_dir = Path("data/raw")
    output_dir = Path("data/processed")
    metadata_dir = Path("data/metadata")
    
    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = []
    for file_path in input_dir.rglob("*"):
        if file_path.suffix.lower() in ['.dcm', '.nii', '.nii.gz']:
            relative_path = file_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix('.nii.gz')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_path.suffix.lower() == '.dcm':
                success = preprocess_dicom(file_path, output_path)
            else:
                success = preprocess_nifti(file_path, output_path)
            
            if success:
                processed_files.append(str(relative_path))
    
    # Save metadata
    with open(metadata_dir / "processed_files.txt", "w") as f:
        f.write("\n".join(processed_files))

with DAG(
    'medical_image_preprocessing',
    default_args=default_args,
    description='Preprocess medical images (DICOM/NIfTI)',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:
    
    process_task = PythonOperator(
        task_id='process_medical_images',
        python_callable=process_files,
        provide_context=True,
    )
    
    process_task 