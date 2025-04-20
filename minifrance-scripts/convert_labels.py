import os
import numpy as np
from PIL import Image
from pathlib import Path

# Define label color mapping (0-15)
label_color_mapping = {
    0: (0, 0, 0),           # No information
    1: (128, 64, 128),      # Urban fabric
    2: (244, 35, 232),      # Industrial, commercial, public, military, private and transport units
    3: (70, 70, 70),        # Mine, dump and construction sites
    4: (102, 102, 156),     # Artificial non-agricultural vegetated areas
    5: (190, 153, 153),     # Arable land (annual crops)
    6: (153, 153, 153),     # Permanent crops
    7: (250, 170, 30),      # Pastures
    8: (220, 220, 0),       # Complex and mixed cultivation patterns
    9: (107, 142, 35),      # Orchards at the fringe of urban classes
    10: (152, 251, 152),    # Forests
    11: (70, 130, 180),     # Herbaceous vegetation associations
    12: (220, 20, 60),      # Open spaces with little or no vegetation
    13: (255, 0, 0),        # Wetlands
    14: (0, 0, 142),        # Water
    15: (220, 220, 220)     # Clouds and shadows
}

# Create reverse mapping from RGB to label index
rgb_to_label = {v: k for k, v in label_color_mapping.items()}

def get_label_from_rgb(rgb_image):
    """Convert RGB label image back to label indices"""
    h, w, _ = rgb_image.shape
    label_array = np.zeros((h, w), dtype=np.uint8)
    for rgb, label in rgb_to_label.items():
        mask = np.all(rgb_image == rgb, axis=-1)
        label_array[mask] = label
    return label_array

def convert_labels(input_dir, output_dir):
    """Convert RGB label images to label indices and save them"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG files in the input directory
    label_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    
    print(f"Found {len(label_files)} label files to convert")
    
    for i, label_file in enumerate(label_files):
        # Read the RGB label image
        rgb_image = np.array(Image.open(os.path.join(input_dir, label_file)))
        
        # Convert to label indices
        label_array = get_label_from_rgb(rgb_image)
        
        # Save as grayscale image
        output_path = os.path.join(output_dir, label_file)
        Image.fromarray(label_array).save(output_path)
        
        if (i + 1) % 100 == 0:
            print(f"Converted {i + 1} labels")

if __name__ == "__main__":
    input_dir = "output2/balanced_patches2/train/labels"
    output_dir = "output2/balanced_patches2/train/label_indices"
    
    print(f"Converting labels from {input_dir} to {output_dir}")
    convert_labels(input_dir, output_dir)
    print("Conversion complete!") 