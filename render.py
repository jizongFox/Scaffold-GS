"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
"""
import open3d as o3d

import json

# Standard library imports
import os
import subprocess
import time
from argparse import ArgumentParser

import numpy as np

# Third-party imports
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, prefilter_voxel, GaussianModel

# Local imports
from scene import Scene
from utils.general_utils import safe_state


def select_available_gpu():
    """Select the GPU with the least memory usage."""
    try:
        cmd = "nvidia-smi -q -d Memory |grep -A4 GPU|grep Used"
        result = (
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
            .stdout.decode()
            .split("\n")
        )
        gpu_id = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Selected GPU: {gpu_id}")
    except Exception as e:
        print(f"GPU selection failed: {e}")
        print("Using default GPU")


def save_image(img, path):
    """Save an image to the specified path, creating directories if necessary."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(img, path)


def render_view(view, gaussians, pipeline, background):
    """Render a single view and measure the rendering time."""
    torch.cuda.synchronize()
    start_time = time.time()

    voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
    render_pkg = render(
        view, gaussians, pipeline, background, visible_mask=voxel_visible_mask
    )

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    return render_pkg["render"], elapsed_time, render_pkg


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    """
    Render a set of views and save the results.

    Args:
        model_path: Path to the model directory
        name: Name of the set (e.g., 'train', 'test')
        iteration: Current iteration number
        views: List of views to render
        gaussians: Gaussian model instance
        pipeline: Pipeline parameters
        background: Background color tensor

    Returns:
        Mean frames per second (FPS) for the rendering
    """
    # Create output directories
    output_dir = os.path.join(model_path, name, f"ours_{iteration}")
    render_path = os.path.join(output_dir, "renders")
    gts_path = os.path.join(output_dir, "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    per_view_dict = {}
    render_times = []
    point_clouds = []
    colors = []

    # Render each view
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} views")):
        rendering, elapsed_time, render_output = render_view(
            view, gaussians, pipeline, background
        )
        render_times.append(elapsed_time)
        point_clouds.append(render_output["xyz"].detach().cpu().numpy())
        colors.append(render_output["color"].detach().cpu().numpy())

        # Get ground truth image
        gt = view.original_image[0:3, :, :]

        # Save rendered and ground truth images
        filename = f"{idx:05d}.png"
        save_image(rendering, os.path.join(render_path, filename))
        save_image(gt, os.path.join(gts_path, filename))

    point_clouds = np.concatenate(point_clouds, axis=0)
    colors = np.concatenate(colors, axis=0)
    # save open3d point cloud
    point_clouds_mean = point_clouds.mean(axis=0)
    norm = np.linalg.norm(point_clouds - point_clouds_mean, axis=1)
    point_clouds = point_clouds[norm < 30]
    colors = colors[norm < 30]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size=1e-6)
    o3d.io.write_point_cloud(os.path.join(output_dir, f"point_cloud_{name}.ply"), pcd)

    # Calculate and print FPS (skip first few frames as warmup)
    if len(render_times) > 5:
        render_times = np.array(render_times[5:])
        fps = 1.0 / render_times.mean()
        print(f"{name.capitalize()} set FPS: \033[1;35m{fps:.5f}\033[0m")
    else:
        fps = 0
        print(f"Not enough views to calculate reliable FPS for {name} set")

    # Save per-view statistics
    stats_path = os.path.join(output_dir, "per_view_count.json")
    with open(stats_path, "w") as fp:
        json.dump(per_view_dict, fp, indent=2)

    return fps


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool = False,
    skip_test: bool = False,
):
    """
    Render both training and test view sets according to specified parameters.

    Args:
        dataset: Model parameters
        iteration: Iteration to load
        pipeline: Pipeline parameters
        skip_train: Whether to skip rendering the training set
        skip_test: Whether to skip rendering the test set
    """
    with torch.no_grad():
        # Initialize the Gaussian model
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
        )

        # Load the scene
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        # Set up background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Ensure model path exists
        os.makedirs(dataset.model_path, exist_ok=True)

        train_fps = test_fps = 0

        # Render training views if requested
        if not skip_train:
            train_fps = render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )

        # Render test views if requested
        if not skip_test:
            test_fps = render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )

        return {
            "train_fps": train_fps,
            "test_fps": test_fps,
            "loaded_iteration": scene.loaded_iter,
        }


def main():
    """Main entry point for the rendering script."""
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to render")
    parser.add_argument(
        "--skip_train", action="store_true", help="Skip rendering training views"
    )
    parser.add_argument(
        "--skip_test", action="store_true", help="Skip rendering test views"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = get_combined_args(parser)

    print(f"Rendering {args.model_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Select GPU with the least memory usage
    select_available_gpu()

    # Render the views
    results = render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
    )

    return results


if __name__ == "__main__":
    main()
