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
# import spaces

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
model = th.compile(model, mode="reduce-overhead", fullgraph=True)
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
    chosen label in the dropdown.
    """
    # Extract label index from dropdown (e.g. "3: (70, 70, 70)" -> 3)
    # label_idx = int(label.split(":")[0])
    
    # Extract label index from dropdown (e.g. "Water: (0, 0, 142)" -> 14)    
    label_idx = int(name_to_label[label.split(":")[0]])
    
    # Get RGB color from dictionary, then convert to hex (#RRGGBB)
    color = label_color_mapping[label_idx]
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    
    # Return an update for the existing ImageEditor component
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
    ), hex_color

tfs_label = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
def copy_output_to_input(output_image):
    return output_image
    
# @spaces.GPU(duration=240)
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

## Batching approach
## 1. Able to generate the inpainted mask with batches
## 2. Need to overlay the original image with the inpainted mask (TODO)
# import torch
# def generate_image(input_image, semantic_drawing, num_imgs: int):
#     """
#     Generate `num_imgs` images with the diffusion model.
#     """
#     # ------------ preprocessing ------------------------------------------------
#     tfm = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3),
#     ])
#     diffusion_steps = 30

#     # (1) RGB image → tensor in [-1,1]
#     img = tfm(input_image).unsqueeze(0)          # (1,3,256,256)  in [-1, 1]

#     # (2) semantic mask → one-hot  (1,19,256,256)
#     raw = semantic_drawing["composite"][:, :, :3]
#     semantic_idx = np.zeros(raw.shape[:2], np.uint8)
#     for k, color in label_color_mapping.items():
#         semantic_idx[(raw == color).all(-1)] = k
#     semantic_idx = torch.from_numpy(semantic_idx).unsqueeze(0).unsqueeze(0)  # (1,1,256,256)

#     one_hot = torch.zeros(1, 19, 256, 256, dtype=torch.float32)
#     one_hot.scatter_(1, semantic_idx.long(), 1.0)

#     # (3) build model kwargs, **repeat along batch dimension**
#     #     so each of the `num_imgs` samples is conditioned the same way.
#     input_semantics = one_hot.repeat(num_imgs, 1, 1, 1)        # (N,19,256,256)
#     model_kwargs = {"y": input_semantics, "s": args.s}

#     # ------------ diffusion ----------------------------------------------------
#     diffusion = recreate_diffusion_with_steps(args, diffusion_steps)

#     sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
#     samples = sample_fn(
#         model,
#         (num_imgs, 3, 256, 256),               # output batch shape
#         clip_denoised=args.clip_denoised,
#         model_kwargs=model_kwargs,
#         progress=True,
#     )                                          # (N,3,256,256)

#     # ------------ post-processing ---------------------------------------------
#     samples = (samples + 1) / 2                                         # back to [0,1]
#     samples = samples.permute(0, 2, 3, 1).cpu().numpy()                 # (N,256,256,3)

#     # return a Python list of individual images (optional but convenient)
#     return [samples[i] for i in range(num_imgs)]

    
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
    gr.Markdown("# Image-to-Image Generation with DDPM")
    
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
                choices=[f"{label_to_name[k]}: {label_color_mapping[k]}" for k in label_color_mapping],
                label="Select Label",
                value=f"No information: {label_to_name[0]}",
                
            )
        with gr.Column(scale=1):
            color_display = gr.ColorPicker(
                label="Selected Color",
                value="#000000",
                interactive=False,
                container=True
            )
    
    # with gr.Row():
    #     prob_mask_slider = gr.Slider(
    #         minimum=0.0,
    #         maximum=1.0,
    #         value=0.5,
    #         step=0.01,
    #         label="Probability Mask",
    #         info="Adjust the probability mask value for image generation"
    #     )
    # with gr.Row():
    #     diffusion_steps_slider = gr.Slider(
    #         minimum=10,
    #         maximum=1000,
    #         step=10,
    #         value=1000,
    #         label="Diffusion Steps",
    #         info="Number of diffusion steps to use"
    #     )
    with gr.Row():
        num_imgs_slider = gr.Slider(
            minimum=1, maximum=4, value=1, step=1,
            label="Images to generate"
        )
    with gr.Row():
        generate_btn = gr.Button("Generate Image", size="large")
       
    
    with gr.Row():
        # output_image = gr.Image(
        #     label="Generated Image",
        #     height=256,
        #     width=256,
        #     container=True,
        #     min_width=400
        # )
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

    
    
    # Connect the dropdown to the update function
    label_dropdown.change(
        fn=update_drawing_color,
        inputs=label_dropdown,
        outputs=[semantic_drawing, color_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10101)
