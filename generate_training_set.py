import os
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    convert_mat_to_png()
