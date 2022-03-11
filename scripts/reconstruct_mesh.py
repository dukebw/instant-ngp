"""CLI commands for meshing NeRF scene reconstructions

Does so by scripting testbed.
Originally based on scripts/run.py.
"""
import json
import logging
import os
from pathlib import Path

import click
import common
import numpy as np

# TODO(brendan): pyngp depends on common being imported first to set up PATH
import pyngp as ngp
from common import ROOT_DIR, write_image
from tqdm import tqdm


@click.group()
def cli():
    """CLI"""


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
    help="Output path for a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format",
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
    help="The scene to load. Can be the scene's name or a full path to the training data",
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
    help="Path to a nerf style transforms.json from which to save screenshots",
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
    render_aabb_min: click.Tuple([float, float, float]),
    render_aabb_max: click.Tuple([float, float, float]),
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
        testbed.compute_and_save_marching_cubes_mesh(
            mesh_output_path, np.tile(marching_cubes_res, 3)
        )

    if ref_transforms is not None:
        testbed.fov_axis = 0
        testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
        ref_transforms["frames"].sort(key=lambda frm: frm["file_path"])
        for frame_idx in range(len(ref_transforms["frames"]) - 1):
            ref_frame = ref_transforms["frames"][frame_idx]
            cam_matrix = ref_frame["transform_matrix"]
            next_cam_matrix = ref_transforms["frames"][frame_idx + 1][
                "transform_matrix"
            ]

            for interpolation_idx in range(n_interpolation):
                alpha = interpolation_idx / n_interpolation
                lerp_cam_matrix = ((1.0 - alpha) * np.array(cam_matrix)) + (
                    alpha * np.array(next_cam_matrix)
                )
                testbed.set_nerf_camera_matrix(lerp_cam_matrix[:-1, :])
                outname = os.path.join(
                    screenshot_dir,
                    Path(ref_frame["file_path"]).stem + f"_{interpolation_idx}.png",
                )

                logging.info(f"Rendering {outname}")
                image = testbed.render(
                    screenshot_width, screenshot_height, screenshot_spp, True
                )
                os.makedirs(os.path.dirname(outname), exist_ok=True)
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
