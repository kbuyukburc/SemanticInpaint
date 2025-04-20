import os
from PIL import Image
import numpy as np

# Define paths for the dataset folders
IMAGES_DIR = "miniFrance/labeled_training/labeled"
LABELS_DIR = "miniFrance/labels"

# Output directories for patches
OUTPUT_IMAGES_DIR = "output2/patches/images"
OUTPUT_LABELS_DIR = "output2/patches/labels"

# Create output directories if they do not exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

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

def convert_label_to_rgb(label_array, mapping):
    """
    Convert a 2D array of label indices into a 3D RGB image using the provided mapping.
    """
    h, w = label_array.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in mapping.items():
        mask = (label_array == label)
        rgb_image[mask] = color
    return rgb_image

def get_base_key(filename):
    """
    Extract the full base key from the filename.
    For example:
      - "06-2014-1015-6335-LA93-0M50-E080.jp2.tif" becomes "06-2014-1015-6335-LA93-0M50-E080"
      - "06-2014-1015-6335-LA93-0M50-E080_UA2012.tif" becomes "06-2014-1015-6335-LA93-0M50-E080"
    """
    base = os.path.basename(filename)
    base_no_ext = base.split('.')[0]  # Remove compound extension
    if "_" in base_no_ext:
        key = base_no_ext.split("_")[0]
    else:
        key = base_no_ext
    return key

# Build a dictionary for label files by base key for fast lookup.
label_files_dict = {}
for root, dirs, files in os.walk(LABELS_DIR):
    for file in files:
        if file.lower().endswith((".tif", ".tiff")):
            key = get_base_key(file)
            full_path = os.path.join(root, file)
            label_files_dict.setdefault(key, []).append(full_path)

# Process each image file in the images directory
patch_size = 1000  # size of the patch (1000 x 1000)

for root, dirs, files in os.walk(IMAGES_DIR):
    for file in files:
        if file.lower().endswith((".tif", ".tiff", ".jp2.tif")):
            image_path = os.path.join(root, file)
            key = get_base_key(file)
            # Check if a corresponding label exists by base key
            if key not in label_files_dict:
                continue  # skip if no matching label file
            
            # For simplicity, take the first matching label file (if there are multiples)
            label_path = label_files_dict[key][0]
            
            # Open the image and label
            try:
                image = Image.open(image_path)
                label = Image.open(label_path)
            except Exception as e:
                print(f"Error opening {image_path} or {label_path}: {e}")
                continue

            # Ensure both image and label have same dimensions
            if image.size != label.size:
                print(f"Size mismatch between image {image_path} and label {label_path}. Skipping.")
                continue
            
            width, height = image.size
            # Split into non-overlapping patches of size patch_size x patch_size
            num_patches_x = width // patch_size
            num_patches_y = height // patch_size
            
            for i in range(num_patches_x):
                for j in range(num_patches_y):
                    left = i * patch_size
                    upper = j * patch_size
                    right = left + patch_size
                    lower = upper + patch_size
                    
                    # Crop the image and label patch
                    image_patch = image.crop((left, upper, right, lower))
                    label_patch = label.crop((left, upper, right, lower))
                    
                    # Convert label patch to a numpy array and then to RGB
                    label_array = np.array(label_patch)
                    
                    # Check if the patch has at least 2 unique labels
                    unique_labels = np.unique(label_array)
                    if len(unique_labels) < 2:
                        continue  # Skip patches with less than 2 unique labels
                    
                    rgb_label_patch = convert_label_to_rgb(label_array, label_color_mapping)
                    rgb_label_patch_img = Image.fromarray(rgb_label_patch)
                    
                    # Generate patch filename using original name and patch indices
                    base_image_name = os.path.splitext(os.path.basename(file))[0]
                    patch_filename = f"{base_image_name}_patch_{i}_{j}.png"
                    
                    # Resize images to 256x256 before saving
                    image_patch_resized = image_patch.resize((256, 256), Image.BILINEAR)
                    rgb_label_patch_img_resized = rgb_label_patch_img.resize((256, 256), Image.NEAREST)
                    
                    # Save the resized image and label patches
                    image_patch_resized.save(os.path.join(OUTPUT_IMAGES_DIR, patch_filename))
                    rgb_label_patch_img_resized.save(os.path.join(OUTPUT_LABELS_DIR, patch_filename))
                    
            print(f"Processed {file} with corresponding label {os.path.basename(label_path)}")
