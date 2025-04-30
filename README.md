---
title: Semantic Inpainting for Satellite Images
emoji: 🎨
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 5.27.0
app_file: app.py
pinned: false
license: unknown
---

# Semantic Inpainting

**Semantic-Aware Image-to-Image Generation Using Diffusion Models for Urban Satellite Imagery**  

A Gradio-based web app that lets you interactively mask and reconstruct regions of high-resolution urban satellite images. Under the hood, a latent diffusion model conditioned on semantic segmentation labels ensures that buildings, roads, forests, and other classes are inpainted in a contextually coherent and visually realistic way.

---

Link to the GitHub repository - [GitHub](https://github.com/kbuyukburc/SemanticInpaint)<br />
Link to the Hugging Face (HF) demo - [Hugging Face](https://huggingface.co/spaces/Kutluhan/SemanticInpaint)

## Features

- **Interactive Gradio GUI**  
  - Upload or select example satellite images  
  - Free-hand drawing mask regions
  - Semantic label selector with color preview  
  - Adjustable guidance scale for faithfulness to labels  
  - Choose number of outputs to generate  
- **High-quality Inpainting**  
  - Latent diffusion conditioned on segmentation maps
  - Supports common urban classes: buildings, roads, vegetation, water, etc.  
- **Flexible Workflow**
  - View original and inpainted images
  - Download and save outputs
  - “Use Output as New Input” button for iterative edits  

---

## Getting Started

### Prerequisites

- Python >= 3.10
- git
- A CUDA-capable GPU (recommended for performance)


### Installation on local computer or using Google Colab or Kaggle with GPU (Preferred)

1. **Clone the HF repository**  
   ```bash
   git clone https://huggingface.co/spaces/Kutluhan/SemanticInpaint
   cd SemanticInpaint
   ```

2. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   pip install gradio
   ```

### Run it directly on Hugging Face Spaces
1. **Visit to HF Spaces page using the [Link](https://huggingface.co/spaces/Kutluhan/SemanticInpaint)**  
   ```bash
   https://huggingface.co/spaces/Kutluhan/SemanticInpaint
   ```

2. **Create an account (requires PRO account for extra Zero-gpu credits) and run it directly (Skip Usage part 1)**  
---

## Usage

1. **Launch the app**  
   ```bash
   python app.py
   ```
   By default, Gradio will run on the public URL.<br />
   You can also select which pre trained model you want to use (default model008000.pt, finetuned on 30 epocs), available [here](https://huggingface.co/Kutluhan/SemanticInpaint/tree/main).<br />
   The pre-trained model can be changed in the app.py file.

2. **In your browser**  
   - **Input Panel**: Upload or pick an urban satellite image from examples.  
   - **Masking & Control Panel**:  
     - Draw to mask regions.  
     - Select semantic class (e.g., Building, Road, Vegetation).  
     - Adjust **Thickness Scale** for how strictly to follow labels.  
     - Choose **Number of Images** to generate.  
   - **Output Panel**:  
     - Compare original v/s. inpainted images.  
     - Download and save the output images or re-feed it for another pass.

---

## Sample Outputs
Here are some sample outputs based on our testing --<br />
- Imagine you want to create new town with wider roads going through a forest<br />
![out1](images/out1.png)<br />

- Imagine you want to create land for permanent crops in a small town<br />
![out2](images/out2.png)<br />

- Imagine you want to create a new city in middle of nowhere that doesn't have any buildings, so you start with building roads<br />
![out3](images/out3.png)<br />

- Imagine you want create land for herbaceous vegetation in a crowded city<br />
![out4](images/out4.png)<br />

- Imagine there was a lot of rain in a particular area leading to water deposits and wetland formation, so you want to create a bridge connecting the two seperated villages<br />
![out5](images/out5.png)<br />




## Recommendations and Insights

### General Recommendations
- **Generate Multiple Samples**: When using the tool, generate 2 to 4 samples to explore different variations. Not all outputs are visually satisfying, so generating multiple samples increases the chance of obtaining a high-quality, contextually appropriate result.

### Label Behavior Insights
- **Multi-Label Combinations That Work Well**
  - **Urban Fabric**: Tends to generate city-like structures and dense urban layouts.
  - **Industrial/Commercial**: Often produces road networks and industrial complexes. Users should preferably draw thin lines to simulate roads when using this label.
  - **Permanent Crops**: Generates brown or green agricultural fields.
  - **Forests**: Creates dense forested regions.
  - **Water**: Primarily generates lakes (greenish color) and occasionally seas (bluish color).
  - **Artificial Non-Agricultural Vegetated Areas**: Complements urban layouts.

- **Labels That Work Well Alone**
  - **Pastures**: Best used independently.
  - **Arable Land**: Works well individually.
  - **Herbaceous Vegetation**: Generates vegetation but sometimes generates white snowy textures.

- **Labels That Do Not Perform Well**
  - **Mine/Dump/Construction**: Limited generation quality due to insufficient training data.
  - **Complex & Mixed Cultivation Patterns**: Inconsistent variation causes inconsistent results.
  - **Orchards**: insufficent training data.
  - **Open Spaces with Little or No Vegetation**: Sparse or incoherent generation.

- **Recommended Combinations**
  - **Wetlands + Water**: Generates more realistic transitions.
  - **Urban Fabric + Industrial/Commercial**: This combination generates city layouts with roads, creating realistic urban environments.

## Best Practices for Using the Tool
- **Use Few Classes at a Time**: Pick 1–2 semantic classes initially, draw them on the mask, and generate images.
- **Iterative Editing**: Select the best image from the generated samples as the new base, then add 1–2 more semantic classes and repeat the generation process.
- **Preserve Context**: Using only 1–2 new mask types at a time helps maintain better visual and semantic context in the generated images.
- **Avoid Large Masks**: Drawing very large masks disrupts spatial coherence. Prefer smaller, targeted edits to preserve surrounding context and improve the realism of generated outputs.
---

