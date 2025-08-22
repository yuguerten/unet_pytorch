import os
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def normalize_image(image_array):
    """Normalize image array to 0-255 range for PNG conversion"""
    # Convert to float64 for processing
    image_array = image_array.astype(np.float64)
    
    # Normalize to 0-255 range
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    if max_val > min_val:
        normalized = ((image_array - min_val) / (max_val - min_val)) * 255
    else:
        normalized = np.zeros_like(image_array)
    
    return normalized.astype(np.uint8)

def convert_mat_to_png():
    """Convert .mat files to PNG images and masks"""
    mat_images_dir = 'mat_dataset'
    
    # Create dataset directories
    dataset_dir = 'dataset'
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    
    create_directory_if_not_exists(dataset_dir)
    create_directory_if_not_exists(images_dir)
    create_directory_if_not_exists(masks_dir)
    
    # Get list of .mat files
    mat_files = [f for f in os.listdir(mat_images_dir) if f.endswith('.mat')]
    mat_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort numerically
    
    print(f"Found {len(mat_files)} .mat files to convert")
    
    converted_count = 0
    failed_count = 0
    
    for mat_file in mat_files:
        try:
            mat_path = os.path.join(mat_images_dir, mat_file)
            base_name = os.path.splitext(mat_file)[0]
            
            # Read the .mat file
            with h5py.File(mat_path, 'r') as f:
                # Extract image and mask
                image_data = np.array(f['cjdata/image']).T  # Transpose for correct orientation
                mask_data = np.array(f['cjdata/tumorMask']).T  # Transpose for correct orientation
                
                # Normalize and convert image
                normalized_image = normalize_image(image_data)
                
                # Convert mask to 0-255 binary
                mask_binary = (mask_data * 255).astype(np.uint8)
                
                # Save as PNG
                image_path = os.path.join(images_dir, f"{base_name}.jpg")
                mask_path = os.path.join(masks_dir, f"{base_name}.jpg")
                
                # Save images using PIL
                Image.fromarray(normalized_image, mode='L').save(image_path)
                Image.fromarray(mask_binary, mode='L').save(mask_path)
                
                converted_count += 1
                
                if converted_count % 100 == 0:
                    print(f"Converted {converted_count}/{len(mat_files)} files")
                    
        except Exception as e:
            print(f"Failed to convert {mat_file}: {str(e)}")
            failed_count += 1
    
    print(f"\nConversion completed!")
    print(f"Successfully converted: {converted_count} files")
    print(f"Failed conversions: {failed_count} files")
    print(f"Images saved in: {images_dir}")
    print(f"Masks saved in: {masks_dir}")

def split_dataset():
    """Split the dataset into train (80%) and validation (20%) sets"""
    dataset_dir = 'dataset'
    images_dir = os.path.join(dataset_dir, 'images')
    masks_dir = os.path.join(dataset_dir, 'masks')
    
    # Create train/val directories
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    
    train_images_dir = os.path.join(train_dir, 'images')
    train_masks_dir = os.path.join(train_dir, 'masks')
    val_images_dir = os.path.join(val_dir, 'images')
    val_masks_dir = os.path.join(val_dir, 'masks')
    
    create_directory_if_not_exists(train_dir)
    create_directory_if_not_exists(val_dir)
    create_directory_if_not_exists(train_images_dir)
    create_directory_if_not_exists(train_masks_dir)
    create_directory_if_not_exists(val_images_dir)
    create_directory_if_not_exists(val_masks_dir)
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort()
    
    print(f"Found {len(image_files)} image files to split")
    
    # Split into train/val (80/20)
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
    
    print(f"Train set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    
    # Move files to respective directories
    import shutil
    
    for file in train_files:
        # Copy image and mask
        shutil.copy2(os.path.join(images_dir, file), os.path.join(train_images_dir, file))
        shutil.copy2(os.path.join(masks_dir, file), os.path.join(train_masks_dir, file))
    
    for file in val_files:
        # Copy image and mask
        shutil.copy2(os.path.join(images_dir, file), os.path.join(val_images_dir, file))
        shutil.copy2(os.path.join(masks_dir, file), os.path.join(val_masks_dir, file))
    
    print(f"Dataset split completed!")
    print(f"Train images: {train_images_dir}")
    print(f"Train masks: {train_masks_dir}")
    print(f"Val images: {val_images_dir}")
    print(f"Val masks: {val_masks_dir}")

if __name__ == "__main__":
    convert_mat_to_png()
    split_dataset()
