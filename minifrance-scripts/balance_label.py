import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import random
import pickle
import argparse

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

def calculate_variance(counts):
    """Calculate variance of pixel counts across classes"""
    values = np.array(list(counts.values()))
    return np.var(values)

def get_patch_score(patch_counts, current_counts):
    """Calculate score for a patch based on how it affects variance"""
    # Create temporary counts including this patch
    temp_counts = defaultdict(int)
    for label in set(patch_counts.keys()) | set(current_counts.keys()):
        temp_counts[label] = current_counts[label] + patch_counts.get(label, 0)
    return -calculate_variance(temp_counts)  # Negative because we want to minimize variance

def clear_directory(directory):
    """Clear all contents of a directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

def analyze_patches(image_files, label_files, cache_file):
    """Analyze patches and cache the results"""
    if os.path.exists(cache_file):
        print(f"Loading cached patch analysis from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("Analyzing patches...")
    all_patches = []
    for i, (img_file, lbl_file) in enumerate(zip(image_files, label_files)):
        src_lbl = os.path.join(ORIGINAL_LABELS_DIR, lbl_file)
        label_img = np.array(Image.open(src_lbl))
        label_indices = get_label_from_rgb(label_img)
        unique_labels, counts = np.unique(label_indices, return_counts=True)
        
        patch_counts = defaultdict(int)
        for label, count in zip(unique_labels, counts):
            patch_counts[label] = count
        
        all_patches.append({
            'image_file': img_file,
            'label_file': lbl_file,
            'counts': patch_counts
        })
        
        if (i + 1) % 100 == 0:
            print(f"Analyzed {i + 1} patches")
    
    print(f"Saving patch analysis to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(all_patches, f)
    
    return all_patches

# Define paths
ORIGINAL_IMAGES_DIR = "output2/patches/images"
ORIGINAL_LABELS_DIR = "output2/patches/labels"
BALANCED_DIR = "output2/balanced_patches2"
BALANCED_IMAGES_DIR = os.path.join(BALANCED_DIR, "images")
BALANCED_LABELS_DIR = os.path.join(BALANCED_DIR, "labels")
CACHE_FILE = "patch_analysis.pkl"

def main():
    parser = argparse.ArgumentParser(description='Balance dataset by minimizing pixel count variance')
    parser.add_argument('--num_frames', type=int, default=3000, help='Number of frames to select')
    parser.add_argument('--remove_background', type=bool, default=True, help='Remove background class')
    args = parser.parse_args()

    # Clear output directories
    print("Clearing output directories...")
    clear_directory(BALANCED_IMAGES_DIR)
    clear_directory(BALANCED_LABELS_DIR)

    # Get list of all patch files
    image_files = sorted([f for f in os.listdir(ORIGINAL_IMAGES_DIR) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(ORIGINAL_LABELS_DIR) if f.endswith('.png')])

    # Ensure we have matching pairs
    assert len(image_files) == len(label_files), "Number of images and labels don't match"
    assert all(img == lbl for img, lbl in zip(image_files, label_files)), "Image and label filenames don't match"

    # Analyze patches (with caching)
    all_patches = analyze_patches(image_files, label_files, CACHE_FILE)
    print(f"\nTotal patches available: {len(all_patches)}")
    
    # Remove patches that have background class (0)
    if args.remove_background:
        filtered_patches = [patch for patch in all_patches if patch['counts'].get(0, 0) == 0]
        all_patches = filtered_patches
        print(f"Patches after removing background class: {len(all_patches)}")

    # Select patches to minimize variance
    selected_patches = []
    current_counts = defaultdict(int)
    TARGET_PATCHES = args.num_frames

    print(f"\nSelecting {TARGET_PATCHES} patches to minimize variance...")
    while len(selected_patches) < TARGET_PATCHES and all_patches:
        # Score remaining patches
        scores = [(i, get_patch_score(patch['counts'], current_counts)) 
                  for i, patch in enumerate(all_patches)]
        
        # Sort by score (descending) and take top 10%
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in scores[:max(1, len(scores) // 10)]]
        
        # Randomly select one from top 10% to avoid getting stuck in local minima
        selected_idx = random.choice(top_indices)
        selected_patch = all_patches.pop(selected_idx)
        selected_patches.append(selected_patch)
        
        # Update current counts
        for label, count in selected_patch['counts'].items():
            current_counts[label] += count
        
        if len(selected_patches) % 100 == 0:
            print(f"Selected {len(selected_patches)} patches")
            print(f"Current variance: {calculate_variance(current_counts):.2f}")
            print(f"Current counts: {current_counts}")

    # Copy selected patches to balanced directory
    print("\nCopying selected patches...")
    for i, patch in enumerate(selected_patches):
        # Copy image
        src_img = os.path.join(ORIGINAL_IMAGES_DIR, patch['image_file'])
        dst_img = os.path.join(BALANCED_IMAGES_DIR, patch['image_file'])
        shutil.copy2(src_img, dst_img)
        
        # Copy label
        src_lbl = os.path.join(ORIGINAL_LABELS_DIR, patch['label_file'])
        dst_lbl = os.path.join(BALANCED_LABELS_DIR, patch['label_file'])
        shutil.copy2(src_lbl, dst_lbl)
        
        if (i + 1) % 100 == 0:
            print(f"Copied {i + 1} patches")

    print("\nFinal Label Pixel Counts:")
    print("------------------")
    for label in sorted(current_counts.keys()):
        count = current_counts[label]
        percentage = (count / sum(current_counts.values())) * 100
        print(f"Label {label:2d}: {count:10,d} pixels ({percentage:6.2f}%)")

    print(f"\nFinal variance: {calculate_variance(current_counts):.2f}")
    print(f"\nTotal pixels: {sum(current_counts.values()):,}")
    print(f"\nSuccessfully created balanced dataset with {len(selected_patches)} pairs")
    print(f"Balanced dataset location: {BALANCED_DIR}")
    print(f"Images: {BALANCED_IMAGES_DIR}")
    print(f"Labels: {BALANCED_LABELS_DIR}")

if __name__ == "__main__":
    main()
