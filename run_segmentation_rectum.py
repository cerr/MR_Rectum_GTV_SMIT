import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from monai import data, transforms
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import Flip
from monai.inferers import sliding_window_inference
from smit_models import smit, configs_smit

from tqdm import tqdm


def get_key_name_part(path, min_len=4):
    """Extract meaningful identifier from file path."""
    fname = os.path.basename(path)
    stem = fname.split('.')[0]
    parent = os.path.basename(os.path.dirname(path))

    if re.fullmatch(r"\d{4,}", stem):
        return stem

    throwaway = {"ctimg", "mrimg", "scan", "img", "image"}
    if len(stem) < min_len or stem.lower() in throwaway:
        return parent

    return stem


def make_seg_filename(path):
    """Generate segmentation filename from input path."""
    key = get_key_name_part(path)
    return f"{key}_seg.nii.gz"


def list_of_strings(arg):
    """Parse comma-separated string into list."""
    return arg.split(',')


def setup_argparser():
    """Setup argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description="Unified SMIT segmentation pipeline with optional TTA and distributed inference")
    
    # Data arguments
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument("--json_list", default=None, type=str, help="dataset json file")
    parser.add_argument("--datasets", default=None, type=list_of_strings, 
                        help="comma-separated list of datasets to pull from json file")
    parser.add_argument("--results_dir", default='results', type=str, help="main output directory")
    parser.add_argument("--output_dir", default=None, type=str, help="secondary output directory")

    # Model arguments
    parser.add_argument("--model_name", default='smit', type=str, help="model name for output directory")
    parser.add_argument("--pretrained_model_path", default=None, type=str, 
                        help="path to pretrained model checkpoint")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--norm_name", default="batch", type=str, help="normalization name")
    parser.add_argument("--use_upernet", action="store_true", help="Use UPERNET decoder")


    # Preprocessing arguments
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    
    # Inference arguments
    parser.add_argument("--infer_overlap", default=0.5, type=float, 
                        help="sliding window inference overlap")
    parser.add_argument("--sw_batch_size", default=16, type=int, 
                        help="sliding window batch size")
    parser.add_argument("--use_tta", action="store_true", 
                        help="use test-time augmentation (horizontal flip)")
    
    # Orientation argument
    parser.add_argument("--skip_orientation", action="store_true",
                        help="skip Orientationd transform (for datasets already in correct orientation)")
    
    # Post-processing arguments
    parser.add_argument("--postproc", default="standard", 
                        choices=['standard', 'conf_threshold'],
                        help="post-processing method: standard (argmax) or conf_threshold")
    parser.add_argument("--conf_threshold", default=0.70, type=float, 
                        help="confidence threshold when using conf_threshold post-processing")

    # Distributed arguments
    parser.add_argument("--distributed", action="store_true", help="use distributed inference")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank of the process")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed training")
    parser.add_argument("--dist_url", default="env://", type=str, help="url for distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

    return parser


def setup_distributed(args):
    """Initialize distributed environment."""
    if args.distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            print("Distributed environment variables not set. Using single GPU.")
            args.distributed = False
            return
        
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()
        
        if args.rank == 0:
            print(f"Distributed initialized: world_size={args.world_size}")


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    """Check if this is the main process."""
    return not args.distributed or args.rank == 0


def create_transforms(args):
    """Create preprocessing and post-processing transforms."""
    
    transform_list = [
        transforms.LoadImaged(keys=["image"]),
        transforms.AddChanneld(keys=["image"]),
    ]
    
    if not args.skip_orientation:
        transform_list.append(
            transforms.Orientationd(keys=["image"], axcodes="RAS")
        )
    
    transform_list.extend([
        transforms.Spacingd(
            keys=["image"], 
            pixdim=(args.space_x, args.space_y, args.space_z), 
            mode=("bilinear")
        ),
        transforms.ScaleIntensityRanged(
            keys=["image"], 
            a_min=args.a_min, a_max=args.a_max, 
            b_min=args.b_min, b_max=args.b_max, 
            clip=True
        ),
        transforms.CropForegroundd(keys=["image"], source_key="image"),
        transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, 0)),
        transforms.SpatialPadd(keys=["image"], spatial_size=(0, 0, args.roi_z), method='end'),
        transforms.ToTensord(keys=["image"]),
    ])
    
    test_transform = transforms.Compose(transform_list)
    
    post_transforms = transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(
            keys="pred",
            transform=test_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])
    
    return test_transform, post_transforms


def load_model(args, device):
    if args.model_name == 'smit':
        config = configs_smit.get_SMIT_128_bias_True_upernet() if args.use_upernet else configs_smit.get_SMIT_128_bias_True()
        model = smit.SMIT_3D_Seg(config,
                                 out_channels=args.out_channels,
                                 img_size=(args.roi_x, args.roi_y, args.roi_z),
                                 norm_name=args.norm_name)
    elif args.model_name == 'swinunetr':
        from models import swin_nvidia
        model = swin_nvidia.SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,  # or add as arg
            norm_name=args.norm_name,
        )
    # rest of checkpoint loading stays identical...
    
    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    model_dict = checkpoint.get("state_dict", checkpoint)
    
    # Handle DDP state dict
    new_state_dict = {}
    for k, v in model_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)
    
    return model


def run_inference_with_tta(model, val_inputs, args):
    """Run inference with optional test-time augmentation."""
    if not args.use_tta:
        pred = sliding_window_inference(
            inputs=val_inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
            mode="gaussian",
        )
        return pred
    
    preds = []
    for do_flip in [False, True]:
        aug_inputs = val_inputs
        
        # Apply flip
        if do_flip:
            aug_inputs = Flip(spatial_axis=1)(aug_inputs)
        
        # Run inference
        pred = sliding_window_inference(
            inputs=aug_inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
            mode="gaussian",
        )
        
        # Invert flip
        if do_flip:
            pred = Flip(spatial_axis=1)(pred)
        
        preds.append(pred)
    
    return torch.mean(torch.stack(preds), dim=0)

def apply_postprocessing(pred_t, args):
    """Apply post-processing to get final segmentation labels."""
    if args.postproc == 'standard':
        pred_labels = torch.argmax(pred_t, dim=0).to(torch.int64)
    else:
        probs = torch.softmax(pred_t, dim=0)
        conf, pred_labels = torch.max(probs, dim=0)
        pred_labels[conf < args.conf_threshold] = 0
        pred_labels = pred_labels.to(torch.int64)
    
    return pred_labels


def process_dataset(dataset_name, args, model, test_transform, post_transforms, device):
    """Process a single dataset."""
    if is_main_process(args):
        print(f'Working on: {dataset_name}')
    
    output_directory = args.output_dir 
    
    if is_main_process(args):
        os.makedirs(output_directory, exist_ok=True)
    
    if args.distributed:
        dist.barrier()
    
    datalist_json = os.path.join(args.data_dir, args.json_list)
    test_files = load_decathlon_datalist(
        datalist_json, 
        True, 
        dataset_name, 
        base_dir=args.data_dir
    )
    
    test_ds = data.Dataset(test_files, transform=test_transform)
    
    # Use distributed sampler if distributed
    if args.distributed:
        sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        sampler = None
    
    test_loader = data.DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False,
        sampler=sampler,
        pin_memory=True
    )
    
    # Progress bar only on main process
    if is_main_process(args):
        loader_iter = tqdm(test_loader, desc=f"{dataset_name}")
    else:
        loader_iter = test_loader
    
    with torch.no_grad():
        for i, batch in enumerate(loader_iter):
            try:
                val_inputs = batch["image"].to(device)
                
                img_name = batch["image_meta_dict"]["filename_or_obj"][0]
                img_name = make_seg_filename(img_name)
                
                batch["pred"] = run_inference_with_tta(model, val_inputs, args)
                
                batch = [post_transforms(i) for i in decollate_batch(batch)]
                
                b0 = batch[0]
                pred_t = b0['pred']
                pred_labels = apply_postprocessing(pred_t, args)
                
                seg_np = pred_labels.cpu().numpy().astype(np.uint8)
                affine = b0["image_meta_dict"].get("original_affine", np.eye(4))
                
                output_path = os.path.join(output_directory, img_name)
                nib.save(nib.Nifti1Image(seg_np, affine), output_path)
                
            except Exception as e:
                print(f"\n[Rank {args.rank}] Error processing {batch['image_meta_dict']['filename_or_obj'][0]}: {e}")
                continue
    
    if args.distributed:
        dist.barrier()
    
    if is_main_process(args):
        print(f"Completed: {dataset_name}\n")


def main():
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Setup distributed
    setup_distributed(args)
    
    # Set device
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process(args):
        print("=" * 60)
        print("Inference Configuration")
        print("=" * 60)
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        print(f"\nDevice: {device}")
        print(f"Distributed: {args.distributed}")
        if args.distributed:
            print(f"World size: {args.world_size}")
        print(f"Orientation transform: {'SKIPPED' if args.skip_orientation else 'RAS'}")
        print(f"TTA: {args.use_tta}")
        print("=" * 60 + "\n")
    
    test_transform, post_transforms = create_transforms(args)
    
    if is_main_process(args):
        print("Loading model...")
    
    model = load_model(args, device)
    
    if is_main_process(args):
        print(f"Model: {args.model_name}")
        print(f"Output channels: {args.out_channels}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Post-processing: {args.postproc}")
        if args.postproc == 'conf_threshold':
            print(f"Confidence threshold: {args.conf_threshold}")
        print()
    
    for dataset_name in args.datasets:
        process_dataset(
            dataset_name, 
            args, 
            model, 
            test_transform, 
            post_transforms, 
            device
        )
    
    cleanup_distributed()
    
    if is_main_process(args):
        print("Inference completed successfully!")


if __name__ == "__main__":
    main()
