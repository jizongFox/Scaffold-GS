from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float, Bool
from torch import Tensor

from scene.cameras import Camera
from scene.gaussian_model import GaussianModel


# @torch.compile
def _world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of world to camera transformation on Gaussians.

    Args:
        means: Gaussian means in world coordinate system. [C, N, 3].
        covars: Gaussian covariances in world coordinate system. [C, N, 3, 3].
        viewmats: world to camera transformation matrices. [C, 4, 4].

    Returns:
        A tuple:

        - **means_c**: Gaussian means in camera coordinate system. [C, N, 3].
        - **covars_c**: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
    """
    R = viewmats[:3, :3]  # [C, 3, 3]
    t = viewmats[:3, 3]  # [C, 3]
    means_c = torch.einsum("ij,nj->ni", R, means) + t[None, :]  # (N, 3)
    covars_c = torch.einsum("ij,njk,lk->nil", R, covars, R)  # [N, 3, 3]
    return means_c, covars_c


def _persp_proj(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    Ks: Tensor,  # [3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """PyTorch implementation of perspective projection for 3D Gaussians.

    Args:
        means: Gaussian means in camera coordinate system. [C, N, 3].
        covars: Gaussian covariances in camera coordinate system. [C, N, 3, 3].
        Ks: Camera intrinsics. [C, 3, 3].
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **means2d**: Projected means. [C, N, 2].
        - **cov2d**: Projected covariances. [C, N, 2, 2].
    """
    means = means[None, ...]
    covars = covars[None, ...]
    Ks = Ks[None, ...]

    C, N, _ = means.shape

    tx, ty, tz = torch.unbind(means, dim=-1)  # [C, N]
    tz2 = tz ** 2  # [C, N]

    fx = Ks[..., 0, 0, None]  # [C, 1]
    fy = Ks[..., 1, 1, None]  # [C, 1]
    cx = Ks[..., 0, 2, None]  # [C, 1]
    cy = Ks[..., 1, 2, None]  # [C, 1]
    tan_fovx = 0.5 * width / fx  # [C, 1]
    tan_fovy = 0.5 * height / fy  # [C, 1]

    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx = tz * torch.clamp(tx / tz, min=-lim_x_neg, max=lim_x_pos)
    ty = tz * torch.clamp(ty / tz, min=-lim_y_neg, max=lim_y_pos)

    O = torch.zeros((C, N), device=means.device, dtype=means.dtype)
    J = torch.stack(
        [fx / tz, O, -fx * tx / tz2, O, fy / tz, -fy * ty / tz2], dim=-1
    ).reshape(C, N, 2, 3)

    cov2d = torch.einsum("...ij,...jk,...kl->...il", J, covars, J.transpose(-1, -2))
    means2d = torch.einsum("cij,cnj->cni", Ks[:, :2, :3], means)  # [C, N, 2]
    means2d = means2d / tz[..., None]  # [C, N, 2]
    return means2d[0], cov2d[0]  # [C, N, 2], [C, N, 2, 2]


def _build_covar3d(covars: Float[Tensor, "n 6"]) -> Float[Tensor, "n 3 3"]:
    cov = torch.zeros((covars.shape[0], 3, 3), device=covars.device, dtype=covars.dtype)
    """
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    """
    cov[:, 0, 0] = covars[:, 0]
    cov[:, 0, 1] = covars[:, 1]
    cov[:, 0, 2] = covars[:, 2]
    cov[:, 1, 1] = covars[:, 3]
    cov[:, 1, 2] = covars[:, 4]
    cov[:, 2, 2] = covars[:, 5]
    return cov


@dataclass(slots=True)
class PrefilterOutput:
    means3d_camera: Float[Tensor, "n 3"]
    depths: Float[Tensor, "n"]
    covars3d_camera: Float[Tensor, "n 3 3"]
    mean2d: Float[Tensor, "n 2"]
    cov2d: Float[Tensor, "n 2 2"]
    radii: Float[Tensor, "n"]
    validate_mask: Bool[Tensor, "n"]


# @torch.compile()


def prefilter_voxel_pytorch(
    viewpoint_cam: "Camera",
    gaussians: "GaussianModel",
    *,
    near_plane: float = 0.2,
    far_plane: float = 1e8,
    eps2d: float = 0.3,
):
    # here I manually code the voxel visibility mask.
    means = gaussians.get_anchor
    covars = _build_covar3d(gaussians.get_covariance())
    viewmats = viewpoint_cam.world2cam
    Ks = viewpoint_cam.intrinsic_matrix
    width = viewpoint_cam.image_width
    height = viewpoint_cam.image_height

    means_c, covars_c = _world_to_cam(means, covars, viewmats)

    means2d, covars2d = _persp_proj(means_c, covars_c, Ks, width, height)

    depths = means_c[..., 2]  # [C, N]

    covars2d = covars2d + torch.eye(2, device=means.device, dtype=means.dtype) * eps2d

    det = (
        covars2d[..., 0, 0] * covars2d[..., 1, 1]
        - covars2d[..., 0, 1] * covars2d[..., 1, 0]
    )
    det = det.clamp(min=1e-10)

    b = (covars2d[..., 0, 0] + covars2d[..., 1, 1]) / 2  # (...,)
    v1 = b + torch.sqrt(torch.clamp(b ** 2 - det, min=0.01))  # (...,)
    radius = torch.ceil(3.0 * torch.sqrt(v1))  # (...,)

    valid = (det > 0) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0
    radii = radius.int()
    validate_mask = radii > 0
    return PrefilterOutput(
        means3d_camera=means_c,
        covars3d_camera=covars_c,
        mean2d=means2d,
        cov2d=covars2d,
        radii=radii,
        validate_mask=validate_mask,
        depths=depths,
    )
