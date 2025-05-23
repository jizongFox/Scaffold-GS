#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from tqdm import tqdm

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(args, id, cam_info, resolution_scale=1.0):
    assert resolution_scale == 1, "Resolution scale should be 1 for now"
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        scale = args.resolution
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution)
        )
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    # print(f'gt_image: {gt_image.shape}')
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    from scene.dataset_readers import CameraInfo

    cam_info: CameraInfo

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        cx=cam_info.cx / scale,
        cy=cam_info.cy / scale,
        fx=cam_info.focal_x / scale,
        fy=cam_info.focal_y / scale,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    assert resolution_scale == 1, "Resolution scale should be 1 for now"

    def process_camera(args_id_cam):
        args, id, cam_info = args_id_cam
        return loadCam(args, id, cam_info, resolution_scale)

    from multiprocessing.dummy import Pool as ThreadPool

    # issue related to https://github.com/pytorch/pytorch/issues/90613
    torch.inverse(torch.ones((1, 1), device="cuda"))

    with ThreadPool() as pool:
        camera_list = [
            x
            for x in tqdm(
                pool.imap(
                    process_camera, [(args, id, c) for id, c in enumerate(cam_infos)]
                ),
                total=len(cam_infos),
                desc="Loading cameras",
            )
        ]
    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
