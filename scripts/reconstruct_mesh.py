"""CLI commands for meshing NeRF scene reconstructions

Does so by scripting testbed.
Originally based on scripts/run.py.
"""
import json
import logging
import os
from pathlib import Path

import click
import numpy as np

# TODO(brendan): pyngp depends on common being imported first to set up PATH
import common
import pyngp as ngp
from common import ROOT_DIR, write_image
from tqdm import tqdm


@click.group()
def cli():
    """CLI"""


@cli.command()
def mesh_from_saved():
    """Mesh from a saved model"""
    # TODO(brendan):
    # 1. load saved model
    # 2. extract vertices of marching cubes mesh using
    #    testbed.compute_marching_cubes_mesh
    # 3. point cloud denoising
    # 4. import the transforms.json from colmap to get camera locations, then
    #    use open3D hidden_point_removal
    # 4. PSR
    # 5. marching cubes


@cli.command()
@click.option(
    "--marching-cubes-res",
    type=int,
    default=256,
    help="Sets the resolution for the marching cubes grid",
)
@click.option(
    "--mesh-output-path",
    type=str,
    default=None,
    help="""
         Output path for a marching-cubes based mesh from the NeRF or SDF model.
         Supports OBJ and PLY format
         """,
)
@click.option(
    "--n-interpolation",
    type=int,
    default=10,
    help="Number of times to interpolate between screenshot transforms keyframes",
)
@click.option(
    "--n-steps",
    type=int,
    default=10 ** 5,
    help="Number of steps to train for before quitting",
)
@click.option(
    "--render-aabb-min",
    nargs=3,
    type=click.Tuple([float, float, float]),
    default=None,
    help="Min values for crop AABB",
)
@click.option(
    "--render-aabb-max",
    type=click.Tuple([float, float, float]),
    default=None,
    help="Max values for crop AABB",
)
@click.option(
    "--scene",
    type=str,
    required=True,
    help="""
         The scene to load.
         Can be the scene's name or a full path to the training data
         """,
)
@click.option(
    "--screenshot-dir",
    type=str,
    default=None,
    help="Which directory to output screenshots to",
)
@click.option("--screenshot-height", type=int, default=1080, help="Screenshot height")
@click.option(
    "--screenshot-spp",
    type=int,
    default=16,
    help="Number of samples per pixel in screenshots",
)
@click.option(
    "--screenshot-transforms",
    type=str,
    default=None,
    help="""
         Path to a nerf style transforms.json from which to save screenshots.
         The format is expected to be that of camera paths saved from the
         testbed GUI.
         """,
)
@click.option("--screenshot-width", type=int, default=1920, help="Screenshot width")
@click.option(
    "--sharpen",
    type=float,
    default=0.0,
    help="Set amount of sharpening applied to NeRF training images",
)
def mesh(
    marching_cubes_res: int,
    mesh_output_path: str,
    n_interpolation: int,
    n_steps: int,
    render_aabb_min: click.Tuple,
    render_aabb_max: click.Tuple,
    scene: str,
    screenshot_dir: str,
    screenshot_height: int,
    screenshot_spp: int,
    screenshot_transforms: str,
    screenshot_width: int,
    sharpen: float,
):
    """Train NeRF and extract a mesh"""
    logging.basicConfig(level=logging.NOTSET)

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.nerf.sharpen = sharpen

    testbed.load_training_data(scene)

    base_network_path = os.path.join(ROOT_DIR, "configs", "nerf", "base.json")
    testbed.reload_network_from_file(base_network_path)

    ref_transforms = None
    if screenshot_transforms is not None:
        logging.info(f"Loading screenshot transforms from {screenshot_transforms}")
        with open(screenshot_transforms, "r", encoding="utf-8") as filestream:
            ref_transforms = json.load(filestream)

    testbed.shall_train = True
    testbed.nerf.render_with_camera_distortion = True
    if render_aabb_min is not None:
        testbed.render_aabb.min = render_aabb_min
    if render_aabb_max is not None:
        testbed.render_aabb.max = render_aabb_max

    previous_training_step = 0
    with tqdm(desc="Training", total=n_steps, unit="step") as progress_bar:
        while testbed.frame() and (testbed.training_step < n_steps):
            progress_bar.update(testbed.training_step - previous_training_step)
            progress_bar.set_postfix(loss=testbed.loss)
            previous_training_step = testbed.training_step

    if mesh_output_path is not None:
        os.makedirs(os.path.dirname(mesh_output_path), exist_ok=True)
        testbed.compute_and_save_marching_cubes_mesh(
            mesh_output_path, np.tile(marching_cubes_res, 3)
        )

    if ref_transforms is not None:
        # NOTE(brendan): based on the batch frame rendering for camera paths
        # described here:
        # https://github.com/NVlabs/instant-ngp/issues/31#issuecomment-1014805317
        testbed.fov_axis = 0
        testbed.fov = ref_transforms["path"][0]["fov"]

        total_interpolated_frames = (
            n_interpolation * (len(ref_transforms["path"]) - 1)
        ) + 1

        testbed.load_camera_path(screenshot_transforms)

        os.makedirs(screenshot_dir, exist_ok=True)
        testbed.camera_smoothing = True
        for frame_idx in range(len(ref_transforms["path"]) - 1):
            for interpolation_idx in range(n_interpolation):
                rendered_idx = (frame_idx * n_interpolation) + interpolation_idx

                outname = os.path.join(screenshot_dir, f"{rendered_idx:04d}.png")

                logging.info(f"Rendering {outname}")
                image = testbed.render(
                    screenshot_width,
                    screenshot_height,
                    screenshot_spp,
                    True,
                    rendered_idx / total_interpolated_frames,
                    (rendered_idx + 1) / total_interpolated_frames,
                    fps=n_interpolation,
                    shutter_fraction=0.5,
                )
                write_image(outname, image)
    else:
        outname = os.path.join(screenshot_dir, Path(base_network_path).stem)
        logging.info(f"Rendering {outname}.png")
        image = testbed.render(
            screenshot_width, screenshot_height, screenshot_spp, True
        )

        os.makedirs(os.path.dirname(outname), exist_ok=True)
        write_image(f"{outname}.png", image)


if __name__ == "__main__":
    cli()
