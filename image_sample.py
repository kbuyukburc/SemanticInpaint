"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import torch as th
import torch.distributed as dist
import torchvision as tv
from guided_diffusion.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

from guided_diffusion.image_datasets import load_data

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

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
label_color_mapping_ts = th.tensor(list(label_color_mapping.values()))
print(label_color_mapping_ts)

def main():
    args = create_argparser().parse_args()
    args.num_classes = args.num_classes + 3 if args.inpainting else args.num_classes
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())

    logger.log("creating data loader...")
    data = load_data(
        dataset_mode=args.dataset_mode,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
        random_crop=False,
        random_flip=False,
        is_train=False
    )

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    image_path = os.path.join(args.results_path, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(args.results_path, 'labels')
    os.makedirs(label_path, exist_ok=True)
    sample_path = os.path.join(args.results_path, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    debug_path = os.path.join(args.results_path, 'debug')
    os.makedirs(debug_path, exist_ok=True)
    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        image = ((batch + 1.0) / 2.0).cuda()
        label = (cond['label_ori'].float() / 255.0).cuda()
        model_kwargs = preprocess_input(batch, cond, num_classes=args.num_classes, inpainting=args.inpainting)
        debug_img = model_kwargs['debug_img']                
        print(cond['path'])

        # set hyperparameter
        model_kwargs['s'] = args.s

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # sample = sample_fn(
        #     model,
        #     (args.batch_size, 3, image.shape[2], image.shape[3]),
        #     clip_denoised=args.clip_denoised,
        #     model_kwargs=model_kwargs,
        #     progress=True
        # )
        # sample = (sample + 1) / 2.0
        sample = th.zeros(1, 3, image.shape[2], image.shape[3]).to(dist_util.dev())

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])
        for j in range(sample.shape[0]):
            print(os.path.join(image_path, cond['path'][j].split('/')[-1]))
            print(os.path.join(sample_path, cond['path'][j].split('/')[-1]))
            print(os.path.join(label_path, cond['path'][j].split('/')[-1]))
            print(os.path.join(debug_path, cond['path'][j].split('/')[-1]))            
            tv.utils.save_image(image[j], os.path.join(image_path, cond['path'][j].split('/')[-1]))
            tv.utils.save_image(sample[j], os.path.join(sample_path, cond['path'][j].split('/')[-1]))
            tv.utils.save_image(label[j], os.path.join(label_path, cond['path'][j].split('/')[-1]))
            tv.utils.save_image(debug_img[j], os.path.join(debug_path, cond['path'][j].split('/')[-1]))            
        logger.log(f"created {len(all_samples) * args.batch_size} samples")
        # break
        if len(all_samples) * args.batch_size > args.num_samples:
            break

    dist.barrier()
    logger.log("sampling complete")


def preprocess_input(batch, data, num_classes, inpainting=False):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    debug_img = th.FloatTensor(bs, h, w, 3).zero_()    
    if inpainting:
        unique_per_batch = [th.unique(x) for x in label_map]
        bypass_label_indices = [th.randperm(len(x))[:random.randint(0, len(x))] for x in unique_per_batch]
        bypass_labels = [x[bypass_label_indices[idx]] for idx, x in enumerate(unique_per_batch)]
        print(unique_per_batch, bypass_labels)
        for idx, bypass_label in enumerate(bypass_labels):
            debug_norm_img = (batch[idx, :, :, :].permute(1, 2, 0) + 1.0) / 2.0
            if random.random() < 0.5:
                bypassed_label = th.where(th.isin(label_map[idx], bypass_label), 0, label_map[idx])
                bypassed_image = th.where(th.isin(label_map[idx], bypass_label), batch[idx], 0) 
                debug_img[idx, :, :, :] = th.where(th.isin(label_map[idx], bypass_label).permute(1,2,0),
                    debug_norm_img, label_color_mapping_ts[label_map[idx].squeeze()]/255.0)
            else:
                mask = get_mask(image_size=(h, w))
                bypassed_label = th.where(mask == 1, label_map[idx], 0)
                bypassed_image = th.where(mask == 1, 0, batch[idx])
                debug_img[idx, :, :, :] = th.where(mask.permute(1,2,0) == 1,                                                   
                                                   label_color_mapping_ts[label_map[idx].squeeze()]/255.0,
                                                   debug_norm_img)
                                                   
            
            label_map[idx, :, :, :] = bypassed_label
            input_label[idx, -3:, :, :] = bypassed_image

    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics, 'debug_img': debug_img.permute(0, 3, 1, 2)}

def get_mask(image_size):
    if random.random() < 0.5:
        regular_mask = bbox2mask(image_size, random_bbox())
        irregular_mask = brush_stroke_mask(image_size, )
        mask = regular_mask | irregular_mask
    else:
        irregular_mask = brush_stroke_mask(image_size, max_loops=10)
        mask = irregular_mask

    return th.from_numpy(mask).permute(2,0,1)


def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


def create_argparser():
    defaults = dict(
        data_dir="",
        dataset_mode="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        model_path="",
        results_path="",
        is_train=False,
        s=1.0,
        inpainting=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
