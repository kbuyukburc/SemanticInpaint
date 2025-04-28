import gradio as gr
import numpy as np
import argparse
import os
import random
import torch as th
th.set_float32_matmul_precision('high')
import torchvision as tv

from guided_diffusion import logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    recreate_diffusion_with_steps,
    add_dict_to_argparser,
    args_to_dict,
)
import argparse

import warnings
from torchvision import transforms
import torch.nn.functional as F
import huggingface_hub
import functools
# import spaces

# Check if running in Hugging Face Spaces
is_spaces = os.environ.get("SPACE_ID") is not None
spaces_decorator = lambda f: f  # Default no-op decorator

if is_spaces:
    try:
        import spaces
        print("Running in Hugging Face Spaces environment")
        # Create a decorator based on FREE_GPU environment variable
        free_gpu = os.environ.get("FREE_GPU", "false").lower() == "true"
        spaces_decorator = functools.partial(spaces.GPU, duration=480, free_gpu=free_gpu)
    except ImportError:
        print("spaces module not found, continuing without it")
        is_spaces = False

model_path = huggingface_hub.hf_hub_download("Kutluhan/SemanticInpaint", "model008000.pt")


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path=model_path,
        results_path="",
        is_train=False,
        s=1.5,
        inpainting=True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    
args = create_argparser().parse_args()
print(args)
def load_model():
    print(args)
    args.num_classes = args.num_classes + 3 if args.inpainting else args.num_classes
    
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to("cuda")
    return model, diffusion

model, diffusion = load_model()
model.convert_to_fp16()
model.eval()

# Check if torch compile should be enabled via environment variable
use_torch_compile = os.environ.get("USE_TORCH_COMPILE", "0").lower() == "1"
if use_torch_compile:
    try:
        print("Enabling PyTorch compilation mode to improve performance")
        model = th.compile(model, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(f"Failed to enable PyTorch compilation mode: {e}")
        print("Continuing without compilation")

# Label to color mapping
label_color_mapping = {
    0: (0, 0, 0),           # No information
    1: (128, 64, 128),      # Urban fabric
    2: (244, 35, 232),      # Industrial / commercial
    3: (70, 70, 70),        # Mine / dump / construction
    4: (102, 102, 156),     # Artificial non-agricultural
    5: (190, 153, 153),     # Arable land
    6: (153, 153, 153),     # Permanent crops
    7: (250, 170, 30),      # Pastures
    8: (220, 220, 0),       # Complex & mixed cultivation
    9: (107, 142, 35),      # Orchards
    10: (152, 251, 152),    # Forests
    11: (70, 130, 180),     # Herbaceous vegetation
    12: (220, 20, 60),      # Open spaces
    13: (255, 0, 0),        # Wetlands
    14: (0, 0, 142),        # Water
    15: (220, 220, 220),    # Clouds & shadows
}

label_to_name = {
    0: "No information",
    1: "Urban fabric",
    2: "Industrial / commercial",
    3: "Mine / dump / construction",
    4: "Artificial non-agricultural",
    5: "Arable land",
    6: "Permanent crops",
    7: "Pastures",
    8: "Complex & mixed cultivation",
    9: "Orchards",
    10: "Forests",
    11: "Herbaceous vegetation",
    12: "Open spaces",
    13: "Wetlands",
    14: "Water",
    15: "Clouds & shadows"
}

name_to_label = {v: k for k, v in label_to_name.items()}

label_color_mapping_ts = th.tensor(list(label_color_mapping.values()))

def create_drawing_canvas():
    """Create a blank white canvas."""
    canvas = np.ones((256, 256, 3), dtype=np.uint8) * 0
    return canvas

def update_drawing_color(label):
    """
    Update the ImageEditor's brush color based on the 
    chosen label in the dropdown and show helpful hints.
    """
    # Extract label index from dropdown (e.g. "Water [USE ALONE]: (0, 0, 142)" -> 14)
    # Handle labels with tags by splitting on ":" and then extracting the base name
    label_part = label.split(":")[0].strip()
    
    # Extract the base label name without tags
    if "[USE ALONE]" in label_part:
        base_label = label_part.replace("[USE ALONE]", "").strip()
    elif "[DONT USE]" in label_part:
        base_label = label_part.replace("[DONT USE]", "").strip()
    else:
        base_label = label_part
    
    # Get the label index
    label_idx = name_to_label.get(base_label, 0)
    
    # Get RGB color from dictionary, then convert to hex (#RRGGBB)
    color = label_color_mapping[label_idx]
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    
    # Define message types and hints for each label
    label_hints = {
        # Multi-Label Combinations That Work Well (info)
        1: ("info", "Urban Fabric: Tends to generate city-like structures and dense urban layouts. Works well with Industrial/Commercial."),
        2: ("info", "Industrial/Commercial: Often produces road networks and industrial complexes. Draw thin lines to simulate roads when using this label."),
        6: ("info", "Permanent Crops: Generates brown or green agricultural fields."),
        10: ("info", "Forests: Creates dense forested regions."),
        14: ("info", "Water: Primarily generates lakes (greenish color) and occasionally seas (bluish color). Combines well with Wetlands."),
        4: ("info", "Artificial Non-Agricultural Vegetated Areas: Complements urban layouts."),
        
        # Labels That Work Well Alone (info)
        7: ("info", "Pastures: Best used independently."),
        5: ("info", "Arable Land: Works well individually."),
        11: ("info", "Herbaceous Vegetation: Sometimes generates white snowy textures. Best used alone."),
        
        # Labels That Do Not Perform Well (warning)
        3: ("warning", "Mine/Dump/Construction: Limited generation quality due to insufficient data."),
        8: ("warning", "Complex & Mixed Cultivation Patterns: Inconsistent results."),
        9: ("warning", "Orchards: Often unreliable."),
        12: ("warning", "Open Spaces with Little or No Vegetation: Sparse or incoherent generation."),
        15: ("warning", "Clouds & Shadows: Limited performance due to insufficient data."),
        
        # Recommended Combinations (info)
        13: ("info", "Wetlands: Works best when combined with Water to generate more realistic transitions."),
        
        # Empty for other labels
        0: ("", ""),
    }
    
    # Get hint for current label
    hint_type, hint_text = label_hints.get(label_idx, ("", ""))
    
    # Return an update for the existing ImageEditor component and the hint
    return gr.ImageEditor(
        label="Semantic Drawing",
        container=True,
        brush=gr.Brush(
            colors=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in label_color_mapping.values()],
            default_color=hex_color,
            color_mode='fixed',
            default_size=20
        ),
        interactive=True
    ), hex_color, hint_type, hint_text

tfs_label = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
def copy_output_to_input(output_image):
    return output_image
    
# Define the generate_image function with conditional decorator
@spaces_decorator
def generate_image(input_image, semantic_drawing, num_imgs):
    """
    Generate image using the model with adjustable prob_mask parameter.
    """    
    print("num_imgs", num_imgs)
    tfs = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    diffusion_steps = 1000
    print(diffusion_steps)
    img = tfs(input_image).unsqueeze(0)
    # Convert semantic_drawing to numpy array if it's not already
    # 256x256x3
    raw_label = semantic_drawing["composite"][:,:,:3].copy()
    # Convert RGB colors to semantic indexes
    semantic_label = np.zeros((raw_label.shape[0], raw_label.shape[1]), dtype=np.uint8)
    for label_idx, color in label_color_mapping.items():
        # Find pixels that match this color exactly
        mask = np.all(raw_label == color, axis=2)
        semantic_label[mask] = label_idx
    
    # Check if any pixels couldn't be mapped to a label
    unmapped = np.all(np.logical_not(np.any(
        [np.all(raw_label == color, axis=2) for color in label_color_mapping.values()],
        axis=0
    )), axis=0)
    
    if np.any(unmapped):
        raise ValueError(f"Found colors in the semantic drawing that don't match any label color")
    semantic_label= semantic_label.reshape(1, 1, 256, 256)
    input_label = th.FloatTensor(1, 19, 256, 256).zero_()    
    input_image = th.tensor(np.where(semantic_label == 0, img, 0))
    input_label[0, -3:, :, :] = input_image
    input_semantics = input_label.scatter_(1, th.tensor(semantic_label).long(), 1.0)
    model_kwargs = {'y': input_semantics.tile(num_imgs, 1, 1, 1),}
    model_kwargs['s'] = args.s
    print(args.s)
    
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    
    output_imgs = []    
    sample = sample_fn(
        model,
        (num_imgs, 3, 256, 256),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        progress=True,
    )
    sample = (sample + 1) / 2.0
    sample = sample.cpu().permute(0, 2, 3, 1)
    for num in range(num_imgs):
        output_img = sample.cpu().numpy()[num]
        output_imgs.append(output_img)
    
    return output_imgs  # return twice: for output and new input

def put_clicked_into_input(evt: gr.SelectData):
    """
    evt.value  -> value associated with the clicked item
                  (a NumPy array *or* a list, gradio versions differ)
    evt.index  -> zero-based index of the thumbnail that was clicked
    """
    clicked = evt.value
    image = clicked['image']['path']
    if isinstance(clicked, list):          # safety for older gradio versions
        clicked = clicked[evt.index]
    return image                         # single image (NumPy array)

def put_selected_into_input(chosen_img):
    print('chosen_img', chosen_img)
    if chosen_img is None:
        raise gr.Error("Please click an image in the gallery first.")
    return chosen_img
    
with gr.Blocks() as demo:
    gr.Markdown("# Satellite Image Semantic Inpainting with DDPM")
    
    # Add recommendations and best practices
    with gr.Accordion("Usage Guidelines & Best Practices", open=True):
        gr.Markdown("""
        ### General Recommendations
        
        - **Generate Multiple Samples**: Generate 2-4 samples to explore different variations. Not all outputs are visually satisfying, so multiple samples increase your chance of getting high-quality results.
        
        ### Best Practices
        
        - **Use Few Classes at a Time**: Start with 1-2 semantic classes, draw them on the mask, and generate images.
        - **Iterative Editing**: Select the best image from the generated samples as your new base, then add 1-2 more semantic classes and repeat.
        - **Preserve Context**: Using only 1-2 new mask types at a time helps maintain better visual and semantic context.
        - **Avoid Large Masks**: Drawing very large masks disrupts spatial coherence. Prefer smaller, targeted edits for better realism.
        
        ### Label Indications
        - Labels marked with **[USE ALONE]** work best independently.
        - Labels marked with **[DONT USE]** generally produce poor results.
        - Unmarked labels can be combined for interesting effects.
        """)
    
    # Create modified label names with tags
    modified_label_names = {}
    for k, v in label_to_name.items():
        if k in [7, 5, 11]:  # Labels that work well alone
            modified_label_names[k] = f"{v} [USE ALONE]"
        elif k in [3, 8, 9, 12, 15]:  # Labels that don't perform well
            modified_label_names[k] = f"{v} [DONT USE]"
        else:
            modified_label_names[k] = v
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Input Image",
                type="numpy",
                container=True,
                height=350
            )
        with gr.Column(scale=1):
            semantic_drawing = gr.ImageEditor(
                label="Semantic Drawing",
                container=True,
                brush=gr.Brush(
                    colors=[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in label_color_mapping.values()],
                    default_color='#000000',
                    color_mode='fixed',
                    default_size=20
                ),
                interactive=True,
                height=350
            )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            label_dropdown = gr.Dropdown(
                choices=[f"{modified_label_names[k]}: {label_color_mapping[k]}" for k in label_color_mapping],
                label="Select Label",
                value=f"{modified_label_names[0]}: {label_color_mapping[0]}",
            )
        with gr.Column(scale=1):
            color_display = gr.ColorPicker(
                label="Selected Color",
                value="#000000",
                interactive=False,
                container=True
            )
    
    with gr.Row():
        num_imgs_slider = gr.Slider(
            minimum=1, maximum=4, value=1, step=1,
            label="Images to generate"
        )
    with gr.Row():
        generate_btn = gr.Button("Generate Image", size="large")
       
    
    with gr.Row():
        output_image = gr.Gallery(
            label="Generated images",
            height="auto", columns=[4],         # 4 columns looks nice up to 8 images
            preview=False,
        )

        selected_img = gr.State(value=None)
    with gr.Row():
        use_as_input_btn = gr.Button("Use Output as New Input")
        
    # Example images
    gr.Examples(
        examples=[
            ["examples/example1.png", create_drawing_canvas()],
            ["examples/example2.png", create_drawing_canvas()],
            ["examples/example3.png", create_drawing_canvas()],
            ["examples/example4.png", "./examples/example4_mask.png"],
            ["examples/example5.png", "./examples/example5_mask.png"],
            ["examples/example6.png", "./examples/example6_mask.png"],
            ["examples/example7.png", "./examples/example7_mask.png"],
        ],
        inputs=[input_image, semantic_drawing]
    )

    # Connect the generate button
    generate_btn.click(
        fn=generate_image,
        inputs=[input_image, semantic_drawing, num_imgs_slider],
        outputs=output_image
    )
    
    output_image.select(
        fn=put_clicked_into_input,
        inputs=None,       # the gallery value is the *selected image*, not the list
        outputs=selected_img        # gr.Image expects a single ndarray → OK
    )

    
    use_as_input_btn.click(
        fn=put_selected_into_input,
        inputs=selected_img,     # comes from our hidden State
        outputs=input_image
    )

    
    
    # Add hint components (hidden from UI but used for handling hints)
    hint_type = gr.Textbox(visible=False)
    hint_text = gr.Textbox(visible=False)

    # Function to display appropriate notification based on hint type and text
    def show_notification(hint_type, hint_text):
        if hint_type == "warning":
            return gr.Warning(hint_text)
        elif hint_type == "info":
            return gr.Info(hint_text)
        return None

    # Connect the dropdown to update functions
    label_dropdown.change(
        fn=update_drawing_color,
        inputs=label_dropdown,
        outputs=[semantic_drawing, color_display, hint_type, hint_text]
    ).then(
        fn=show_notification,
        inputs=[hint_type, hint_text],
        outputs=None
    )

if __name__ == "__main__":
    demo.launch()
