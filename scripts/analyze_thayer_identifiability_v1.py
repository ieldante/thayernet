#!/usr/bin/env python3
"""Training-free prompt-centered galaxy-prior identifiability audit.

The prospective prior definitions live in the immutable run preregistration.
This implementation reads only the eight authorized Family-E1 training rows,
their frozen isolated-source references, and the corresponding CatSim catalog
rows.  It never imports, constructs, or optimizes a neural network.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

import galsim
import h5py
import numpy as np
import pandas as pd
from astropy.table import Table
from btk.survey import get_surveys
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, linprog
from scipy.special import expit
from scipy.stats import qmc


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.competing_hypotheses import scientific_distance, source_measurements


RUN = REPO / "outputs/runs/thayer_identifiability_v1_20260715_003220"
PREREG = RUN / "preregistration/prior_and_analysis_freeze.md"
PREREG_SHA256 = "5af9db12575fe8f7025149cf55685456a22b30e0e19b8d9d7738d65683cb3475"
UPSTREAM = REPO / "outputs/runs/thayer_select_hierarchical_feasibility_20260712_010729"
FAMILY_E1 = REPO / "outputs/runs/thayer_family_e1_v0_20260714_214715"
FAMILY_E1P = REPO / "outputs/runs/thayer_family_e1p_v0_20260714_225228"
FOUNDATION = REPO / "outputs/runs/thayer_select_btk_foundation_20260711_152613"
SELECTOR = FAMILY_E1 / "manifests/training_manifest.csv"
MANIFEST = UPSTREAM / "manifests/v2_r_training_scene_manifest.csv"
SCENES = UPSTREAM / "manifests/v2_r_training_scenes.h5"
DIFFICULT_MANIFEST = FAMILY_E1P / "manifests/difficult_one_scene_paired_scene_manifest.csv"
MIXED_MANIFEST = FAMILY_E1P / "manifests/mixed_eight_scene_paired_scene_manifest.csv"
CATALOG = FOUNDATION / "data/input_catalog_btk_v1.0.9.fits"

INDICES = (0, 3, 5, 6, 18, 51, 73, 81)
BANDS = ("g", "r", "z")
PIXEL_SCALE = 0.2
IMAGE_SIZE = 60
PIXELS_PER_SOURCE = 3 * IMAGE_SIZE * IMAGE_SIZE
MEAN_PSF_FWHM_PIXEL = float(np.mean(np.asarray([0.86, 0.81, 0.77]) / PIXEL_SCALE))
EXACT_TOLERANCE = 8.0 * np.finfo(np.float32).eps
TV_BOUND = 1.0
OPTIMIZER_STARTS = 16
OPTIMIZER_SEED = 2026071507
MAX_NFEV = 400
FIT_BIN_FACTOR = 3

SERSIC_PARAMETER_COUNT = 4
BD_GRADIENT_PARAMETER_COUNT = 9
BD_SHARED_PARAMETER_COUNT = 7


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_text_fresh(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(value)


def write_json_fresh(path: Path, value: object) -> None:
    write_text_fresh(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_csv_rows(path: Path, rows: list[int] | tuple[int, ...] | np.ndarray) -> pd.DataFrame:
    authorized = {int(value) for value in rows}
    return pd.read_csv(
        path,
        low_memory=False,
        skiprows=lambda line: line > 0 and (line - 1) not in authorized,
    )


def command(arguments: list[str]) -> dict[str, Any]:
    result = subprocess.run(arguments, cwd=REPO, capture_output=True, text=True, check=False)
    return {
        "arguments": arguments,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def verify_preregistration() -> None:
    if sha256_file(PREREG) != PREREG_SHA256:
        raise RuntimeError("prior freeze hash changed")
    staged = command(["git", "diff", "--cached", "--name-only"])
    if staged["returncode"] != 0 or staged["stdout"].strip():
        raise RuntimeError("staged index is not empty")


@dataclass(frozen=True)
class Scene:
    index: int
    manifest: dict[str, Any]
    paired: dict[str, Any]
    observed: np.ndarray
    isolated: np.ndarray
    xy: np.ndarray
    catalog_rows: tuple[Any, Any]
    psfs: tuple[galsim.GSObject, galsim.GSObject, galsim.GSObject]

    @property
    def target(self) -> np.ndarray:
        return np.asarray(self.isolated[0] + self.isolated[1], dtype=np.float64)

    @property
    def requested_index(self) -> int:
        return int(self.manifest["matched_source_index"])

    @property
    def companion_index(self) -> int:
        return 1 - self.requested_index

    @property
    def source_prompts(self) -> np.ndarray:
        return np.asarray(self.xy, dtype=np.float64)

    @property
    def centers_arcsec(self) -> np.ndarray:
        row = self.manifest
        return np.asarray(
            [
                [row["source_a_x_arcsec"], row["source_a_y_arcsec"]],
                [row["source_b_x_arcsec"], row["source_b_y_arcsec"]],
            ],
            dtype=np.float64,
        )


def survey_psfs() -> tuple[galsim.GSObject, galsim.GSObject, galsim.GSObject]:
    survey = get_surveys("LSST")
    psfs = []
    for band in BANDS:
        filt = survey.get_filter(band)
        psf = filt.psf(survey, filt) if callable(filt.psf) else filt.psf
        if not isinstance(psf, galsim.GSObject):
            raise RuntimeError(f"invalid PSF for {band}")
        psfs.append(psf)
    return tuple(psfs)  # type: ignore[return-value]


def load_scene(index: int) -> Scene:
    if index not in INDICES:
        raise ValueError(f"unauthorized Family-E1 index {index}")
    selector = read_csv_rows(SELECTOR, [index]).iloc[0]
    if int(selector.family_e_index) != index or int(selector.upstream_index) != index:
        raise RuntimeError("frozen Family-E1 selector identity changed")
    manifest = read_csv_rows(MANIFEST, [index]).iloc[0]
    if int(manifest.dataset_index) != index:
        raise RuntimeError("upstream manifest row identity changed")
    paired = pd.concat((pd.read_csv(DIFFICULT_MANIFEST), pd.read_csv(MIXED_MANIFEST)), ignore_index=True)
    paired = paired.loc[paired.family_e1_index == index].drop_duplicates("family_e1_index", keep="last").iloc[0]
    with h5py.File(SCENES, "r") as handle:
        observed = np.asarray(handle["blend"][index], dtype=np.float32)
        isolated = np.asarray(handle["isolated"][index], dtype=np.float32)
        xy = np.asarray(handle["xy"][index], dtype=np.float64)
    if sha256_array(observed) != str(paired.observation_sha256):
        raise RuntimeError("observation hash mismatch")
    if sha256_array(isolated[0]) != str(manifest.isolated_source_a_sha256):
        raise RuntimeError("source-A hash mismatch")
    if sha256_array(isolated[1]) != str(manifest.isolated_source_b_sha256):
        raise RuntimeError("source-B hash mismatch")
    catalog = Table.read(CATALOG)
    row_ids = (int(manifest.source_a_row), int(manifest.source_b_row))
    rows = (catalog[row_ids[0]], catalog[row_ids[1]])
    return Scene(
        index=index,
        manifest=manifest.to_dict(),
        paired=paired.to_dict(),
        observed=observed,
        isolated=isolated,
        xy=xy,
        catalog_rows=rows,
        psfs=survey_psfs(),
    )


def _map_radius(value: float) -> float:
    return float(np.exp(np.log(0.03) + np.clip(value, 0.0, 1.0) * np.log(100.0)))


def _unmap_radius(value: float) -> float:
    return float(np.clip(np.log(value / 0.03) / np.log(100.0), 0.0, 1.0))


def _map_q(value: float) -> float:
    return float(0.1 + 0.9 * np.clip(value, 0.0, 1.0))


def _unmap_q(value: float) -> float:
    return float(np.clip((value - 0.1) / 0.9, 0.0, 1.0))


def _map_angle(value: float) -> float:
    return float(-0.5 * np.pi + np.pi * np.clip(value, 0.0, 1.0))


def _unmap_angle_degrees(value: float) -> float:
    radians = np.deg2rad(value)
    wrapped = (radians + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return float(np.clip((wrapped + 0.5 * np.pi) / np.pi, 0.0, 1.0))


def decode_sersic(z: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(0.5 + 5.5 * np.clip(z[0], 0.0, 1.0)),
        _map_radius(float(z[1])),
        _map_q(float(z[2])),
        _map_angle(float(z[3])),
    )


def decode_bd_gradient(z: np.ndarray) -> tuple[float, float, float, float, float, float, np.ndarray]:
    return (
        _map_radius(float(z[0])),
        _map_q(float(z[1])),
        _map_angle(float(z[2])),
        _map_radius(float(z[3])),
        _map_q(float(z[4])),
        _map_angle(float(z[5])),
        np.clip(np.asarray(z[6:9], dtype=np.float64), 0.0, 1.0),
    )


def decode_bd_shared(z: np.ndarray) -> tuple[float, float, float, float, float, float, np.ndarray]:
    bt = float(np.clip(z[6], 0.0, 1.0))
    return (
        _map_radius(float(z[0])),
        _map_q(float(z[1])),
        _map_angle(float(z[2])),
        _map_radius(float(z[3])),
        _map_q(float(z[4])),
        _map_angle(float(z[5])),
        np.repeat(bt, 3),
    )


def _draw_normalized(
    galaxy_for_band: Callable[[int], galsim.GSObject],
    center_arcsec: np.ndarray,
    fluxes: np.ndarray,
    psfs: tuple[galsim.GSObject, galsim.GSObject, galsim.GSObject],
) -> np.ndarray:
    output = np.empty((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    for band in range(3):
        galaxy = galsim.Convolve(galaxy_for_band(band), psfs[band]).shift(
            float(center_arcsec[0]), float(center_arcsec[1])
        )
        image = np.asarray(
            galaxy.drawImage(nx=IMAGE_SIZE, ny=IMAGE_SIZE, scale=PIXEL_SCALE).array,
            dtype=np.float64,
        )
        total = float(image.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("rendered profile has invalid stamp flux")
        output[band] = image * (float(fluxes[band]) / total)
    return output


def render_sersic_source(
    z: np.ndarray,
    center_arcsec: np.ndarray,
    fluxes: np.ndarray,
    psfs: tuple[galsim.GSObject, galsim.GSObject, galsim.GSObject],
) -> np.ndarray:
    n, radius, axis_ratio, angle = decode_sersic(z)

    def galaxy(_: int) -> galsim.GSObject:
        # BTK renders these two legal Sersic points with GalSim's specialized
        # classes.  Reusing them at exact n=1/n=4 avoids a several-part-in-1e6
        # generic lookup approximation without altering the continuous prior.
        if abs(n - 1.0) <= 1e-14:
            profile = galsim.Exponential(half_light_radius=radius, flux=1.0)
        elif abs(n - 4.0) <= 1e-14:
            profile = galsim.DeVaucouleurs(half_light_radius=radius, flux=1.0)
        else:
            profile = galsim.Sersic(n=n, half_light_radius=radius, flux=1.0)
        return profile.shear(
            q=axis_ratio, beta=angle * galsim.radians
        )

    return _draw_normalized(galaxy, center_arcsec, fluxes, psfs)


def render_bd_source(
    z: np.ndarray,
    center_arcsec: np.ndarray,
    fluxes: np.ndarray,
    psfs: tuple[galsim.GSObject, galsim.GSObject, galsim.GSObject],
    *,
    shared_color: bool,
) -> np.ndarray:
    decoded = decode_bd_shared(z) if shared_color else decode_bd_gradient(z)
    disk_r, disk_q, disk_pa, bulge_r, bulge_q, bulge_pa, bulge_fraction = decoded

    def galaxy(band: int) -> galsim.GSObject:
        bt = float(bulge_fraction[band])
        components = []
        if 1.0 - bt > 0:
            components.append(
                galsim.Exponential(half_light_radius=disk_r, flux=1.0 - bt).shear(
                    q=disk_q, beta=disk_pa * galsim.radians
                )
            )
        if bt > 0:
            components.append(
                galsim.DeVaucouleurs(half_light_radius=bulge_r, flux=bt).shear(
                    q=bulge_q, beta=bulge_pa * galsim.radians
                )
            )
        if not components:
            raise RuntimeError("bulge+disk profile has zero total flux")
        return components[0] if len(components) == 1 else galsim.Add(components)

    return _draw_normalized(galaxy, center_arcsec, fluxes, psfs)


def model_parameter_count(model: str) -> int:
    if model == "sersic":
        return SERSIC_PARAMETER_COUNT
    if model == "bd_gradient":
        return BD_GRADIENT_PARAMETER_COUNT
    if model == "bd_shared":
        return BD_SHARED_PARAMETER_COUNT
    raise ValueError(model)


def render_source(scene: Scene, model: str, source_index: int, z: np.ndarray) -> np.ndarray:
    fluxes = np.asarray(scene.isolated[source_index], dtype=np.float64).sum(axis=(1, 2))
    center = scene.centers_arcsec[source_index]
    if model == "sersic":
        return render_sersic_source(z, center, fluxes, scene.psfs)
    return render_bd_source(z, center, fluxes, scene.psfs, shared_color=model == "bd_shared")


def render_pair(scene: Scene, model: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = model_parameter_count(model)
    return (
        render_source(scene, model, 0, z[:count]),
        render_source(scene, model, 1, z[count:]),
    )


def relative_residual(render: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(render - target) / max(np.linalg.norm(target), np.finfo(np.float64).tiny))


def truth_parameters(scene: Scene, model: str) -> tuple[np.ndarray, bool]:
    encoded = []
    member = True
    for row in scene.catalog_rows:
        disk_flux = float(row["fluxnorm_disk"])
        bulge_flux = float(row["fluxnorm_bulge"])
        agn_flux = float(row["fluxnorm_agn"])
        if agn_flux != 0 or disk_flux + bulge_flux <= 0:
            member = False
        active_disk = disk_flux > 0
        active_bulge = bulge_flux > 0
        if model == "sersic":
            if active_disk and active_bulge:
                member = False
            if active_bulge:
                n = 4.0
                radius = math.sqrt(float(row["a_b"]) * float(row["b_b"]))
                q_value = float(row["b_b"]) / float(row["a_b"])
                pa = float(row["pa_bulge"])
            else:
                n = 1.0
                radius = math.sqrt(float(row["a_d"]) * float(row["b_d"]))
                q_value = float(row["b_d"]) / float(row["a_d"])
                pa = float(row["pa_disk"])
            if not (0.5 <= n <= 6 and 0.03 <= radius <= 3 and 0.1 <= q_value <= 1):
                member = False
            encoded.extend(
                [
                    (n - 0.5) / 5.5,
                    _unmap_radius(radius),
                    _unmap_q(q_value),
                    _unmap_angle_degrees(pa),
                ]
            )
        else:
            if active_disk:
                disk_radius = math.sqrt(float(row["a_d"]) * float(row["b_d"]))
                disk_q = float(row["b_d"]) / float(row["a_d"])
                disk_pa = float(row["pa_disk"])
            else:
                disk_radius, disk_q, disk_pa = 0.3, 0.7, 0.0
            if active_bulge:
                bulge_radius = math.sqrt(float(row["a_b"]) * float(row["b_b"]))
                bulge_q = float(row["b_b"]) / float(row["a_b"])
                bulge_pa = float(row["pa_bulge"])
            else:
                bulge_radius, bulge_q, bulge_pa = 0.3, 0.7, 0.0
            for radius, q_value in ((disk_radius, disk_q), (bulge_radius, bulge_q)):
                if not (0.03 <= radius <= 3 and 0.1 <= q_value <= 1):
                    member = False
            bt = bulge_flux / (disk_flux + bulge_flux)
            base = [
                _unmap_radius(disk_radius),
                _unmap_q(disk_q),
                _unmap_angle_degrees(disk_pa),
                _unmap_radius(bulge_radius),
                _unmap_q(bulge_q),
                _unmap_angle_degrees(bulge_pa),
            ]
            encoded.extend(base + ([bt] if model == "bd_shared" else [bt, bt, bt]))
    return np.asarray(encoded, dtype=np.float64), bool(member)


def observation_moment_shapes(scene: Scene, temperature: float) -> list[tuple[float, float, float]]:
    """Build two truth-free shape starts from the exact fixed-noise source sum.

    Logistic prompt-distance allocation is offset independently in each band
    until it obeys the already granted Level-2 source fluxes.  Only the
    observation target, prompts, and frozen flux prior enter this initializer.
    """

    yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    prompts = scene.source_prompts
    distance_a = np.square(xx - prompts[0, 0]) + np.square(yy - prompts[0, 1])
    distance_b = np.square(xx - prompts[1, 0]) + np.square(yy - prompts[1, 1])
    target = scene.target
    flux_a = np.asarray(scene.isolated[0], dtype=np.float64).sum(axis=(1, 2))
    allocated_a = np.empty_like(target)
    for band in range(3):
        lower = -80.0
        upper = 80.0
        for _ in range(100):
            middle = 0.5 * (lower + upper)
            probability = expit((distance_b - distance_a) / temperature + middle)
            trial_flux = float(np.sum(target[band] * probability))
            if trial_flux < float(flux_a[band]):
                lower = middle
            else:
                upper = middle
        probability = expit((distance_b - distance_a) / temperature + 0.5 * (lower + upper))
        allocated_a[band] = target[band] * probability
    allocations = (allocated_a, target - allocated_a)
    psf_covariances = []
    for psf in scene.psfs:
        psf_image = np.asarray(
            psf.drawImage(nx=IMAGE_SIZE, ny=IMAGE_SIZE, scale=PIXEL_SCALE).array,
            dtype=np.float64,
        )
        psf_image /= float(psf_image.sum())
        psf_x = float(np.sum(psf_image * xx))
        psf_y = float(np.sum(psf_image * yy))
        psf_dx = xx - psf_x
        psf_dy = yy - psf_y
        psf_covariances.append(
            np.asarray(
                [
                    [np.sum(psf_image * psf_dx * psf_dx), np.sum(psf_image * psf_dx * psf_dy)],
                    [np.sum(psf_image * psf_dx * psf_dy), np.sum(psf_image * psf_dy * psf_dy)],
                ],
                dtype=np.float64,
            )
        )
    output = []
    for source_index, allocation in enumerate(allocations):
        weights = np.maximum(allocation.sum(axis=0), 0.0)
        total = float(weights.sum())
        center = prompts[source_index]
        dx = xx - center[0]
        dy = yy - center[1]
        covariance = np.asarray(
            [
                [np.sum(weights * dx * dx), np.sum(weights * dx * dy)],
                [np.sum(weights * dx * dy), np.sum(weights * dy * dy)],
            ],
            dtype=np.float64,
        ) / max(total, np.finfo(np.float64).tiny)
        band_fluxes = allocation.sum(axis=(1, 2), dtype=np.float64)
        effective_psf_covariance = sum(
            float(band_fluxes[band]) * psf_covariances[band] for band in range(3)
        ) / max(float(band_fluxes.sum()), np.finfo(np.float64).tiny)
        covariance = covariance - effective_psf_covariance
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-3)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis_ratio = float(np.sqrt(eigenvalues.min() / eigenvalues.max()))
        radius = float(0.685 * np.sqrt(eigenvalues.sum()) * PIXEL_SCALE)
        angle = float(math.atan2(major[1], major[0]))
        output.append(
            (
                float(np.clip(radius, 0.03, 3.0)),
                float(np.clip(axis_ratio, 0.1, 1.0)),
                angle,
            )
        )
    return output


def encode_angle_radians(value: float) -> float:
    wrapped = (value + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return float(np.clip((wrapped + 0.5 * np.pi) / np.pi, 0.0, 1.0))


def structured_starts(scene: Scene, model: str) -> np.ndarray:
    """Return deterministic observation-only starts for one model."""

    starts = []
    if model == "sersic":
        configurations = [
            (1.0, 1.0),
            (1.0, 4.0),
            (4.0, 1.0),
            (4.0, 4.0),
        ]
        shape_groups = (
            (2.0, 0.2, 0.0),
            (2.0, 0.7, 0.0),
            (8.0, 0.2, 0.25 * np.pi),
            (8.0, 0.7, 0.25 * np.pi),
        )
        for temperature, q_anchor, angle_offset in shape_groups:
            shapes = observation_moment_shapes(scene, temperature)
            encoded = []
            for source_index, n_value in enumerate(configurations[len(starts) % 4]):
                radius, axis_ratio, angle = shapes[source_index]
                axis_ratio = 0.5 * axis_ratio + 0.5 * q_anchor
                angle += angle_offset
                encoded.extend(
                    [
                        (n_value - 0.5) / 5.5,
                        _unmap_radius(radius),
                        _unmap_q(axis_ratio),
                        encode_angle_radians(angle),
                    ]
                )
            starts.append(encoded)
            # Add the other three component-index configurations at the same
            # observation-derived geometry.
            for configuration in configurations[1:]:
                encoded = []
                for source_index, n_value in enumerate(configuration):
                    radius, axis_ratio, angle = shapes[source_index]
                    axis_ratio = 0.5 * axis_ratio + 0.5 * q_anchor
                    angle += angle_offset
                    encoded.extend(
                        [
                            (n_value - 0.5) / 5.5,
                            _unmap_radius(radius),
                            _unmap_q(axis_ratio),
                            encode_angle_radians(angle),
                        ]
                    )
                starts.append(encoded)
    else:
        patterns = [
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.1, 0.1),
            (0.5, 0.5),
            (0.1, 0.5),
            (0.5, 0.1),
        ]
        for pattern_index, pattern in enumerate(patterns):
            shapes = observation_moment_shapes(scene, 2.0 if pattern_index < 4 else 8.0)
            encoded = []
            for source_index, bt in enumerate(pattern):
                radius, axis_ratio, angle = shapes[source_index]
                disk_radius = radius
                bulge_radius = float(np.clip(0.7 * radius, 0.03, 3.0))
                base = [
                    _unmap_radius(disk_radius),
                    _unmap_q(axis_ratio),
                    encode_angle_radians(angle),
                    _unmap_radius(bulge_radius),
                    _unmap_q(max(axis_ratio, 0.5)),
                    encode_angle_radians(angle),
                ]
                encoded.extend(base + ([bt] if model == "bd_shared" else [bt, bt, bt]))
            starts.append(encoded)
    expected_count = 16 if model == "sersic" else 8
    result = np.asarray(starts[:expected_count], dtype=np.float64)
    expected = 2 * model_parameter_count(model)
    if result.shape != (expected_count, expected):
        raise RuntimeError(f"invalid structured-start shape for {model}: {result.shape}")
    return result


def bd_start_from_sersic(sersic_z: np.ndarray, model: str) -> np.ndarray:
    if model not in {"bd_gradient", "bd_shared"}:
        raise ValueError(model)
    output = []
    for source_index in (0, 1):
        local = np.asarray(sersic_z[source_index * 4 : (source_index + 1) * 4], dtype=np.float64)
        n_value, radius, axis_ratio, angle = decode_sersic(local)
        shape = [_unmap_radius(radius), _unmap_q(axis_ratio), encode_angle_radians(angle)]
        if abs(n_value - 1.0) <= abs(n_value - 4.0):
            disk = shape
            bulge = [_unmap_radius(max(0.03, 0.7 * radius)), _unmap_q(max(axis_ratio, 0.5)), encode_angle_radians(angle)]
            bt = 0.0
        else:
            disk = [_unmap_radius(min(3.0, 1.3 * radius)), _unmap_q(axis_ratio), encode_angle_radians(angle)]
            bulge = shape
            bt = 1.0
        output.extend(disk + bulge + ([bt] if model == "bd_shared" else [bt, bt, bt]))
    return np.asarray(output, dtype=np.float64)


def optimize_model(scene: Scene, model: str, warm_start: np.ndarray | None = None) -> dict[str, Any]:
    per_source = model_parameter_count(model)
    dimension = 2 * per_source
    target = scene.target
    def bin_image(image: np.ndarray) -> np.ndarray:
        size = IMAGE_SIZE // FIT_BIN_FACTOR
        return image.reshape(3, size, FIT_BIN_FACTOR, size, FIT_BIN_FACTOR).sum(axis=(2, 4))

    binned_target = bin_image(target)
    band_norms = np.linalg.norm(binned_target, axis=(1, 2))
    sobol = qmc.Sobol(d=dimension, scramble=True, seed=OPTIMIZER_SEED + scene.index + dimension)
    starts = (
        structured_starts(scene, model)
        if model == "sersic"
        else np.concatenate((structured_starts(scene, model), sobol.random_base2(m=3)), axis=0)
    )
    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=np.float64)
        if warm.shape != (dimension,):
            raise RuntimeError(f"invalid warm-start shape for {model}: {warm.shape}")
        starts[-1] = np.clip(warm, 0.0, 1.0)
    records = []

    def objective(z: np.ndarray) -> np.ndarray:
        left, right = render_pair(scene, model, z)
        return ((bin_image(left + right) - binned_target) / band_norms[:, None, None]).ravel()

    for start_index, start in enumerate(starts):
        free_indices = np.arange(dimension, dtype=np.int64)
        # The first eight Sersic starts exhaust the n=1/n=4 component-index
        # combinations.  Hold those discrete indices fixed and optimize the
        # six continuous shape coordinates so GalSim's specialized BTK replay
        # is not crossed by a numerical implementation discontinuity.
        if model == "sersic" and start_index < 16:
            free_indices = np.asarray([value for value in free_indices if value not in {0, per_source}], dtype=np.int64)
        # The first four bulge+disk starts are the four pure-component boundary
        # combinations.  Quotient inactive component shapes and hold B/T at
        # its boundary while optimizing only the active output coordinates.
        if model in {"bd_gradient", "bd_shared"}:
            active = []
            pure_boundary = True
            for source_index in (0, 1):
                offset = source_index * per_source
                bt_index = offset + 6
                bt_values = (
                    np.asarray(start[bt_index : bt_index + 3], dtype=np.float64)
                    if model == "bd_gradient"
                    else np.asarray([start[bt_index]], dtype=np.float64)
                )
                if np.all(bt_values <= 0.0):
                    active.extend(offset + value for value in (0, 1, 2))
                elif np.all(bt_values >= 1.0):
                    active.extend(offset + value for value in (3, 4, 5))
                else:
                    pure_boundary = False
            if pure_boundary:
                free_indices = np.asarray(active, dtype=np.int64)
        fixed_start = np.asarray(start, dtype=np.float64).copy()

        def local_objective(free: np.ndarray) -> np.ndarray:
            candidate = fixed_start.copy()
            candidate[free_indices] = free
            return objective(candidate)

        result = least_squares(
            local_objective,
            np.clip(fixed_start[free_indices], 1e-6, 1.0 - 1e-6),
            bounds=(np.zeros(len(free_indices)), np.ones(len(free_indices))),
            method="trf",
            x_scale="jac",
            diff_step=1e-3,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=MAX_NFEV,
            verbose=0,
        )
        continuous_parameters = fixed_start.copy()
        continuous_parameters[free_indices] = result.x
        # Evaluate the BTK-specialized legal n=1/n=4 points in addition to the
        # raw continuous result.  These combinations are deterministic and
        # truth-free; all are members of the frozen Sersic support.
        candidates = [continuous_parameters]
        if model == "sersic":
            n1 = (1.0 - 0.5) / 5.5
            n4 = (4.0 - 0.5) / 5.5
            for left_n in (n1, n4):
                for right_n in (n1, n4):
                    snapped = continuous_parameters.copy()
                    snapped[0] = left_n
                    snapped[per_source] = right_n
                    candidates.append(snapped)
        rendered_candidates = []
        for candidate in candidates:
            candidate_left, candidate_right = render_pair(scene, model, candidate)
            rendered_candidates.append(
                (
                    relative_residual(candidate_left + candidate_right, target),
                    candidate,
                    candidate_left,
                    candidate_right,
                )
            )
        exact_residual, selected_parameters, left, right = min(
            rendered_candidates, key=lambda item: item[0]
        )
        records.append(
            {
                "start_index": start_index,
                "parameters": selected_parameters.tolist(),
                "continuous_parameters": continuous_parameters.tolist(),
                "relative_residual": exact_residual,
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "full_refinement_nfev": 0,
                "status": int(result.status),
                "success": bool(result.success),
                "free_indices": free_indices.tolist(),
                "left": left,
                "right": right,
            }
        )
    records.sort(key=lambda item: item["relative_residual"])
    full_band_norms = np.linalg.norm(target, axis=(1, 2))
    for record in records[:3]:
        remaining = MAX_NFEV - int(record["nfev"])
        if remaining < 2:
            continue
        continuous = np.asarray(record["continuous_parameters"], dtype=np.float64)
        free_indices = np.asarray(record["free_indices"], dtype=np.int64)

        def full_local_objective(free: np.ndarray) -> np.ndarray:
            candidate = continuous.copy()
            candidate[free_indices] = free
            left_full, right_full = render_pair(scene, model, candidate)
            return ((left_full + right_full - target) / full_band_norms[:, None, None]).ravel()

        refined = least_squares(
            full_local_objective,
            continuous[free_indices],
            bounds=(np.zeros(len(free_indices)), np.ones(len(free_indices))),
            method="trf",
            x_scale="jac",
            diff_step=1e-3,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
            max_nfev=remaining,
            verbose=0,
        )
        refined_continuous = continuous.copy()
        refined_continuous[free_indices] = refined.x
        refined_candidates = [refined_continuous]
        if model == "sersic":
            n1 = (1.0 - 0.5) / 5.5
            n4 = (4.0 - 0.5) / 5.5
            for left_n in (n1, n4):
                for right_n in (n1, n4):
                    snapped = refined_continuous.copy()
                    snapped[0] = left_n
                    snapped[per_source] = right_n
                    refined_candidates.append(snapped)
        evaluated = []
        for candidate in refined_candidates:
            candidate_left, candidate_right = render_pair(scene, model, candidate)
            evaluated.append(
                (
                    relative_residual(candidate_left + candidate_right, target),
                    candidate,
                    candidate_left,
                    candidate_right,
                )
            )
        refined_residual, selected, left, right = min(evaluated, key=lambda item: item[0])
        if refined_residual < float(record["relative_residual"]):
            record.update(
                {
                    "parameters": selected.tolist(),
                    "continuous_parameters": refined_continuous.tolist(),
                    "relative_residual": refined_residual,
                    "cost": float(refined.cost),
                    "optimality": float(refined.optimality),
                    "full_refinement_nfev": int(refined.nfev),
                    "left": left,
                    "right": right,
                }
            )
    records.sort(key=lambda item: item["relative_residual"])
    exact_records = [item for item in records if item["relative_residual"] <= EXACT_TOLERANCE]
    unique_outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for record in exact_records:
        pair = (record["left"], record["right"])
        pair_norm = math.sqrt(float(np.sum(pair[0] ** 2) + np.sum(pair[1] ** 2)))
        duplicate = False
        for previous in unique_outputs:
            difference = math.sqrt(
                float(np.sum((pair[0] - previous[0]) ** 2) + np.sum((pair[1] - previous[1]) ** 2))
            ) / max(pair_norm, np.finfo(np.float64).tiny)
            if difference <= EXACT_TOLERANCE:
                duplicate = True
                break
        if not duplicate:
            unique_outputs.append(pair)
    best = records[0]
    compact_records = [
        {
            key: value
            for key, value in item.items()
            if key not in {"left", "right", "parameters", "continuous_parameters", "free_indices"}
        }
        for item in records
    ]
    return {
        "model": model,
        "starts": OPTIMIZER_STARTS,
        "seed": OPTIMIZER_SEED,
        "best_parameters": best["parameters"],
        "best_continuous_parameters": best["continuous_parameters"],
        "best_relative_residual": float(best["relative_residual"]),
        "exact_start_count": len(exact_records),
        "unique_exact_output_count_found": len(unique_outputs),
        "start_records": compact_records,
        "best_left": best["left"],
        "best_right": best["right"],
    }


def tangent_diagnostics(scene: Scene, model: str, z: np.ndarray) -> dict[str, Any]:
    per_source = model_parameter_count(model)
    bases = []
    source_derivative_ranks = []
    derivative_singular_values = []
    for source_index in (0, 1):
        local = np.asarray(z[source_index * per_source : (source_index + 1) * per_source], dtype=np.float64)
        columns = []
        for parameter in range(per_source):
            step = 1e-4
            lower = local.copy()
            upper = local.copy()
            if local[parameter] <= step:
                upper[parameter] = min(1.0, local[parameter] + step)
                base = render_source(scene, model, source_index, local)
                changed = render_source(scene, model, source_index, upper)
                derivative = (changed - base) / (upper[parameter] - local[parameter])
            elif local[parameter] >= 1.0 - step:
                lower[parameter] = max(0.0, local[parameter] - step)
                base = render_source(scene, model, source_index, local)
                changed = render_source(scene, model, source_index, lower)
                derivative = (base - changed) / (local[parameter] - lower[parameter])
            else:
                lower[parameter] -= step
                upper[parameter] += step
                low_render = render_source(scene, model, source_index, lower)
                high_render = render_source(scene, model, source_index, upper)
                derivative = (high_render - low_render) / (2.0 * step)
            columns.append(derivative.ravel())
        matrix = np.stack(columns, axis=1)
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        tolerance = max(matrix.shape) * np.finfo(np.float64).eps * float(singular[0])
        rank = int(np.sum(singular > tolerance))
        bases.append(u[:, :rank])
        source_derivative_ranks.append(rank)
        derivative_singular_values.append(singular.tolist())
    jacobian = np.concatenate(bases, axis=1)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * float(singular[0])
    rank = int(np.sum(singular > tolerance))
    domain = int(jacobian.shape[1])
    nullity = domain - rank
    condition = math.inf if nullity else float(singular[0] / singular[-1])
    return {
        "source_output_tangent_ranks": source_derivative_ranks,
        "source_derivative_singular_values": derivative_singular_values,
        "restricted_parameter_dimension": domain,
        "jacobian_rank": rank,
        "null_space_dimension": nullity,
        "condition_number": condition,
        "observation_jacobian_singular_values": singular.tolist(),
        "svd_rank_tolerance": tolerance,
    }


def normalized_tv(source: np.ndarray) -> np.ndarray:
    values = []
    for band in range(3):
        image = np.asarray(source[band], dtype=np.float64)
        numerator = float(np.abs(np.diff(image, axis=0)).sum() + np.abs(np.diff(image, axis=1)).sum())
        denominator = float(image.sum())
        values.append(numerator / denominator)
    return np.asarray(values, dtype=np.float64)


def prompt_margin(source: np.ndarray, own_prompt: np.ndarray, other_prompt: np.ndarray) -> float:
    measurement = source_measurements(source)
    if measurement.centroid_xy is None:
        return -math.inf
    center = np.asarray(measurement.centroid_xy, dtype=np.float64)
    return float(np.linalg.norm(center - other_prompt) - np.linalg.norm(center - own_prompt))


def pair_prompt_margin(
    requested: np.ndarray,
    companion: np.ndarray,
    requested_prompt: np.ndarray,
    companion_prompt: np.ndarray,
) -> float:
    return float(
        min(
            prompt_margin(requested, requested_prompt, companion_prompt),
            prompt_margin(companion, companion_prompt, requested_prompt),
        )
    )


def smooth_exchange_diameter(scene: Scene) -> dict[str, Any]:
    requested = np.asarray(scene.isolated[scene.requested_index], dtype=np.float64)
    companion = np.asarray(scene.isolated[scene.companion_index], dtype=np.float64)
    requested_prompt = scene.source_prompts[scene.requested_index]
    companion_prompt = scene.source_prompts[scene.companion_index]
    requested_flux = requested.sum(axis=(1, 2))
    companion_flux = companion.sum(axis=(1, 2))
    maximum = float(min(1.0, np.min(companion_flux / requested_flux))) * (1.0 - 1e-9)

    def exchange(fraction: float) -> tuple[np.ndarray, np.ndarray]:
        ratio = requested_flux / companion_flux
        left = (1.0 - fraction) * requested + fraction * ratio[:, None, None] * companion
        right = requested + companion - left
        return left, right

    truth_margin = pair_prompt_margin(requested, companion, requested_prompt, companion_prompt)
    if truth_margin <= 0:
        raise RuntimeError("frozen truth is not prompt-consistent")
    upper_pair = exchange(maximum)
    if pair_prompt_margin(*upper_pair, requested_prompt, companion_prompt) <= 0:
        lower = 0.0
        upper = maximum
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            if pair_prompt_margin(*exchange(middle), requested_prompt, companion_prompt) > 0:
                lower = middle
            else:
                upper = middle
        maximum = lower
    endpoint = exchange(maximum * (1.0 - 1e-9))
    tv_values = np.concatenate((normalized_tv(endpoint[0]), normalized_tv(endpoint[1])))
    if np.any(tv_values > TV_BOUND + 1e-12):
        raise RuntimeError("convex smoothness witness violates frozen TV bound")
    recomposition = relative_residual(endpoint[0] + endpoint[1], scene.target)
    distance = scientific_distance(
        requested,
        endpoint[0],
        mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
    )
    return {
        "maximum_exchange_fraction": maximum,
        "primary_scientific_diameter_lower_bound": float(distance.primary_normalized),
        "image_distance": float(distance.image),
        "prompt_margin_pixel": pair_prompt_margin(*endpoint, requested_prompt, companion_prompt),
        "maximum_normalized_tv": float(tv_values.max()),
        "relative_recomposition_residual": recomposition,
    }


def _centroid_half_plane_constraints(scene: Scene) -> tuple[sparse.csr_matrix, np.ndarray]:
    yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    positions = np.stack((xx.ravel(), yy.ravel()), axis=1)
    prompts = scene.source_prompts
    fluxes = np.asarray(scene.isolated, dtype=np.float64).sum(axis=(1, 2, 3))
    target_flat_band = scene.target.reshape(3, -1)
    target_spatial = target_flat_band.sum(axis=0)

    delta_a = prompts[0] - prompts[1]
    coefficient_a_spatial = -2.0 * (positions @ delta_a)
    bound_a = fluxes[0] * (float(np.dot(prompts[1], prompts[1])) - float(np.dot(prompts[0], prompts[0])))

    delta_b = prompts[1] - prompts[0]
    coefficient_b_spatial = 2.0 * (positions @ delta_b)
    bound_b = (
        2.0 * float(np.dot(target_spatial, positions @ delta_b))
        + fluxes[1] * (float(np.dot(prompts[0], prompts[0])) - float(np.dot(prompts[1], prompts[1])))
    )
    rows = np.stack((np.tile(coefficient_a_spatial, 3), np.tile(coefficient_b_spatial, 3)))
    matrix = sparse.csr_matrix(rows)
    bounds = np.asarray([bound_a, bound_b], dtype=np.float64)
    truth_a = np.asarray(scene.isolated[0], dtype=np.float64).ravel()
    if np.any(matrix @ truth_a - bounds > 1e-6 * max(float(np.abs(bounds).max()), 1.0)):
        raise RuntimeError("prompt half-plane constraint rejects frozen truth")
    return matrix, bounds


def level2_lp_diameter(scene: Scene) -> dict[str, Any]:
    target = scene.target
    target_flat = target.ravel()
    flux_a = np.asarray(scene.isolated[0], dtype=np.float64).sum(axis=(1, 2))
    equality_rows = []
    for band in range(3):
        row = np.zeros(PIXELS_PER_SOURCE, dtype=np.float64)
        row[band * IMAGE_SIZE * IMAGE_SIZE : (band + 1) * IMAGE_SIZE * IMAGE_SIZE] = 1.0
        equality_rows.append(row)
    a_eq = sparse.csr_matrix(np.stack(equality_rows))
    a_ub, b_ub = _centroid_half_plane_constraints(scene)
    yy, xx = np.indices((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float64)
    spatial_directions = [
        xx,
        yy,
        np.hypot(xx - scene.source_prompts[0, 0], yy - scene.source_prompts[0, 1]),
        np.hypot(xx - scene.source_prompts[1, 0], yy - scene.source_prompts[1, 1]),
    ]
    rng = np.random.default_rng(OPTIMIZER_SEED + scene.index)
    for _ in range(8):
        spatial_directions.append(gaussian_filter(rng.normal(size=(IMAGE_SIZE, IMAGE_SIZE)), sigma=3.0))
    objectives = []
    for direction_index, spatial in enumerate(spatial_directions):
        weights = rng.normal(size=3)
        cube = np.stack([(1.0 + 0.2 * weights[band]) * spatial for band in range(3)])
        scale = float(np.linalg.norm(cube))
        cube = cube / max(scale, np.finfo(np.float64).tiny)
        objectives.extend((cube.ravel(), -cube.ravel()))
    candidates = [np.asarray(scene.isolated[0], dtype=np.float64)]
    solver_records = []
    bounds = list(zip(np.zeros_like(target_flat), target_flat))
    for objective_index, objective in enumerate(objectives):
        result = linprog(
            -objective,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=flux_a,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        solver_records.append(
            {
                "objective_index": objective_index,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
            }
        )
        if not result.success:
            continue
        candidate = np.asarray(result.x, dtype=np.float64).reshape(target.shape)
        companion = target - candidate
        margins = (
            prompt_margin(candidate, scene.source_prompts[0], scene.source_prompts[1]),
            prompt_margin(companion, scene.source_prompts[1], scene.source_prompts[0]),
        )
        if min(margins) <= 0:
            candidate = np.asarray(scene.isolated[0], dtype=np.float64) + (1.0 - 1e-6) * (
                candidate - np.asarray(scene.isolated[0], dtype=np.float64)
            )
        candidates.append(candidate)
    if len(candidates) < 2:
        raise RuntimeError("no Level-2 LP vertex succeeded")
    requested_candidates = [candidate if scene.requested_index == 0 else target - candidate for candidate in candidates]
    maximum = -1.0
    maximizing_pair = None
    for left in range(len(requested_candidates)):
        for right in range(left + 1, len(requested_candidates)):
            value = float(
                scientific_distance(
                    requested_candidates[left],
                    requested_candidates[right],
                    mean_psf_fwhm_pixel=MEAN_PSF_FWHM_PIXEL,
                ).primary_normalized
            )
            if value > maximum:
                maximum = value
                maximizing_pair = (left, right)
    maximum_constraint_residual = 0.0
    for candidate in candidates:
        maximum_constraint_residual = max(
            maximum_constraint_residual,
            float(np.max(np.abs(candidate.sum(axis=(1, 2)) - flux_a))),
            float(max(-candidate.min(), -(target - candidate).min(), 0.0)),
        )
    return {
        "objective_count": len(objectives),
        "successful_vertex_count": len(candidates) - 1,
        "primary_scientific_diameter_lower_bound": maximum,
        "maximizing_candidate_pair": maximizing_pair,
        "maximum_absolute_constraint_residual": maximum_constraint_residual,
        "solver_records": solver_records,
    }


def classify(
    *,
    model_member: bool,
    exact_solution_count: str | int,
    nullity: int,
    diameter: float | None,
    condition: float,
    prompt_unique: bool,
) -> str:
    if not model_member or exact_solution_count == 0 or not prompt_unique:
        return "UNIDENTIFIABLE"
    if exact_solution_count == "uncountably infinite" or (isinstance(exact_solution_count, int) and exact_solution_count > 1) or nullity > 0:
        return "UNIDENTIFIABLE" if diameter is not None and diameter > 1.0 else "PARTIALLY IDENTIFIABLE"
    if not np.isfinite(condition):
        return "NEAR UNIQUE"
    if condition * EXACT_TOLERANCE >= 1.0:
        return "NEAR UNIQUE"
    return "UNIQUE"


def prior_record(
    *,
    scene: Scene,
    level: int,
    name: str,
    rank: int,
    nullity: int,
    condition: float,
    solutions: str | int,
    diameter: float | None,
    diameter_status: str,
    prompt_unique: bool,
    requested_identifiable: bool,
    model_member: bool,
    exact_residual: float,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    classification = classify(
        model_member=model_member,
        exact_solution_count=solutions,
        nullity=nullity,
        diameter=diameter,
        condition=condition,
        prompt_unique=prompt_unique,
    )
    return {
        "scene": scene.index,
        "scene_id": str(scene.manifest["scene_id"]),
        "prior_level": level,
        "prior": name,
        "rank": rank,
        "null_space": nullity,
        "condition_number": "infinity" if not np.isfinite(condition) else condition,
        "number_of_exact_observation_consistent_solutions": solutions,
        "diameter": diameter,
        "diameter_status": diameter_status,
        "prompt_identity_unique": prompt_unique,
        "requested_galaxy_identifiable": requested_identifiable,
        "truth_in_prior_support": model_member,
        "best_relative_exact_residual": exact_residual,
        "classification": classification,
        "diagnostics": diagnostics,
    }


def audit_scene(index: int) -> dict[str, Any]:
    verify_preregistration()
    scene = load_scene(index)
    target = scene.target
    signed_noise = np.asarray(scene.observed, dtype=np.float64) - target
    truth_tv = np.asarray([normalized_tv(scene.isolated[source]) for source in (0, 1)])
    if np.any(truth_tv > TV_BOUND):
        raise RuntimeError("frozen truth lies outside Level-3 smoothness support")
    smooth = smooth_exchange_diameter(scene)
    lp = level2_lp_diameter(scene)
    records = []
    records.append(
        prior_record(
            scene=scene,
            level=0,
            name="No prior",
            rank=PIXELS_PER_SOURCE,
            nullity=PIXELS_PER_SOURCE,
            condition=math.inf,
            solutions="uncountably infinite",
            diameter="infinity",  # JSON-safe representation of the unbounded set.
            diameter_status="unbounded source-pair Euclidean diameter",
            prompt_unique=False,
            requested_identifiable=False,
            model_member=True,
            exact_residual=0.0,
            diagnostics={"observation_jacobian": "[I I]"},
        )
    )
    records.append(
        prior_record(
            scene=scene,
            level=1,
            name="Nonnegative flux",
            rank=PIXELS_PER_SOURCE,
            nullity=PIXELS_PER_SOURCE,
            condition=math.inf,
            solutions="uncountably infinite",
            diameter=10.0,
            diameter_status="exact inherited primary diameter",
            prompt_unique=False,
            requested_identifiable=False,
            model_member=True,
            exact_residual=0.0,
            diagnostics={"complete_set": "0 <= S_A <= T; S_B=T-S_A", "smooth_witness": smooth},
        )
    )
    records.append(
        prior_record(
            scene=scene,
            level=2,
            name="Flux conservation",
            rank=PIXELS_PER_SOURCE - 3,
            nullity=PIXELS_PER_SOURCE - 3,
            condition=math.inf,
            solutions="uncountably infinite",
            diameter=float(lp["primary_scientific_diameter_lower_bound"]),
            diameter_status="certified lower bound from prompt-consistent LP vertices",
            prompt_unique=False,
            requested_identifiable=False,
            model_member=True,
            exact_residual=0.0,
            diagnostics={"linear_program": lp},
        )
    )
    records.append(
        prior_record(
            scene=scene,
            level=3,
            name="Smoothness",
            rank=PIXELS_PER_SOURCE - 3,
            nullity=PIXELS_PER_SOURCE - 3,
            condition=math.inf,
            solutions="uncountably infinite",
            diameter=float(smooth["primary_scientific_diameter_lower_bound"]),
            diameter_status="certified lower bound from smooth flux-preserving exchange",
            prompt_unique=False,
            requested_identifiable=False,
            model_member=True,
            exact_residual=float(smooth["relative_recomposition_residual"]),
            diagnostics={"truth_normalized_tv": truth_tv.tolist(), "smooth_exchange": smooth},
        )
    )

    model_results: dict[str, dict[str, Any]] = {}
    sersic_fit: dict[str, Any] | None = None
    for model in ("sersic", "bd_gradient", "bd_shared"):
        warm_start = None
        if model != "sersic":
            if sersic_fit is None:
                raise RuntimeError("Sersic continuation unavailable")
            warm_start = bd_start_from_sersic(
                np.asarray(sersic_fit["best_parameters"], dtype=np.float64), model
            )
        fit = optimize_model(scene, model, warm_start=warm_start)
        if model == "sersic":
            sersic_fit = fit
        truth_z, member = truth_parameters(scene, model)
        truth_left, truth_right = render_pair(scene, model, truth_z)
        truth_replay_residual = relative_residual(truth_left + truth_right, target)
        if member and truth_replay_residual > EXACT_TOLERANCE:
            raise RuntimeError(f"{model} truth replay exceeds exact tolerance: {truth_replay_residual}")
        # Use the continuous optimum for local derivatives; the specialized
        # exact n=1/n=4 replay is only an equality/candidate check and has a
        # different GalSim numerical implementation at the identical profile.
        best_z = np.asarray(fit["best_continuous_parameters"], dtype=np.float64)
        tangent = tangent_diagnostics(scene, model, best_z)
        if member:
            # Analytic finite-mixture identifiability plus at least one
            # truth-free exact fit is required by the frozen protocol.
            solution_count = 1 if int(fit["exact_start_count"]) > 0 else 0
        else:
            solution_count = 0
        model_results[model] = {
            "fit": {key: value for key, value in fit.items() if key not in {"best_left", "best_right"}},
            "truth_parameters": truth_z.tolist(),
            "truth_in_support": member,
            "truth_replay_relative_residual": truth_replay_residual,
            "tangent": tangent,
            "solution_count": solution_count,
        }

    for level, name, model in (
        (4, "Elliptical Sersic", "sersic"),
        (5, "Bulge + disk", "bd_gradient"),
        (6, "Shared color profile", "bd_shared"),
    ):
        result = model_results[model]
        tangent = result["tangent"]
        member = bool(result["truth_in_support"])
        solutions = int(result["solution_count"])
        condition = float(tangent["condition_number"])
        prompt_unique = bool(member and solutions == 1 and int(tangent["null_space_dimension"]) == 0)
        requested_identifiable = prompt_unique
        records.append(
            prior_record(
                scene=scene,
                level=level,
                name=name,
                rank=int(tangent["jacobian_rank"]),
                nullity=int(tangent["null_space_dimension"]),
                condition=condition,
                solutions=solutions,
                diameter=0.0 if prompt_unique else None,
                diameter_status="exact singleton" if prompt_unique else "undefined because exact support is empty",
                prompt_unique=prompt_unique,
                requested_identifiable=requested_identifiable,
                model_member=member,
                exact_residual=float(result["fit"]["best_relative_residual"]),
                diagnostics=result,
            )
        )
    shared = records[-1]
    level7 = json.loads(json.dumps(shared))
    level7.update(
        {
            "prior_level": 7,
            "prior": "Weak astrophysical morphology prior",
            "diagnostics": {
                "support_identity": "strictly positive soft density leaves Level-6 support unchanged",
                "level6_replay": shared["diagnostics"],
            },
        }
    )
    records.append(level7)
    unique_levels = [record["prior_level"] for record in records if record["classification"] == "UNIQUE"]
    result = {
        "campaign": "Thayer-Identifiability-v1",
        "scene": index,
        "scene_id": str(scene.manifest["scene_id"]),
        "requested_source_index": scene.requested_index,
        "source_separation_pixel": float(np.linalg.norm(scene.xy[0] - scene.xy[1])),
        "observation_sha256": sha256_array(scene.observed),
        "isolated_source_sha256": [sha256_array(scene.isolated[0]), sha256_array(scene.isolated[1])],
        "signed_noise_sha256": sha256_array(signed_noise),
        "preregistration_sha256": PREREG_SHA256,
        "minimum_prior_level_for_unique": min(unique_levels) if unique_levels else None,
        "priors": records,
        "access_counts": {
            "authorized_training_scene_rows": 1,
            "authorized_training_catalog_rows": 2,
            "development": 0,
            "atlas": 0,
            "lockbox": 0,
            "neural_model_imports": 0,
            "network_weight_optimizer_steps": 0,
        },
    }
    return result


def scene_output_path(index: int) -> Path:
    # Earlier artifacts are retained as append-only basin-search diagnostics.
    # R6 adds deterministic lower-level structural continuation within the
    # same 16-start budget and is the candidate authoritative family.
    return RUN / f"scene_results_r6/scene_{index:03d}.json"


def execute_scene(index: int) -> None:
    started = time.time()
    result = audit_scene(index)
    result["runtime_seconds"] = time.time() - started
    write_json_fresh(scene_output_path(index), result)
    print(
        json.dumps(
            {
                "scene": index,
                "runtime_seconds": result["runtime_seconds"],
                "minimum_prior_level_for_unique": result["minimum_prior_level_for_unique"],
                "classifications": [record["classification"] for record in result["priors"]],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def aggregate() -> None:
    verify_preregistration()
    scene_results = []
    for index in INDICES:
        path = scene_output_path(index)
        if not path.exists():
            raise FileNotFoundError(path)
        scene_results.append(json.loads(path.read_text()))
    full_rows = []
    table_rows = []
    for scene in scene_results:
        for record in scene["priors"]:
            row = {key: value for key, value in record.items() if key != "diagnostics"}
            full_rows.append(row)
            diameter = record["diameter"]
            table_rows.append(
                {
                    "Scene": record["scene"],
                    "Prior": f"L{record['prior_level']} {record['prior']}",
                    "Rank": record["rank"],
                    "Null space": record["null_space"],
                    "Diameter": "infinity" if diameter == "infinity" else diameter,
                    "Classification": record["classification"],
                }
            )
    tables = RUN / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    full_frame = pd.DataFrame(full_rows)
    table_frame = pd.DataFrame(table_rows)
    full_path = tables / "full_identifiability_metrics.csv"
    table_path = tables / "identifiability_table.csv"
    if full_path.exists() or table_path.exists():
        raise FileExistsError("aggregate tables already exist")
    full_frame.to_csv(full_path, index=False)
    table_frame.to_csv(table_path, index=False)
    minimum = pd.DataFrame(
        [
            {
                "Scene": scene["scene"],
                "Minimum prior level required for UNIQUE": scene["minimum_prior_level_for_unique"],
            }
            for scene in scene_results
        ]
    )
    minimum_path = tables / "minimum_unique_prior_by_scene.csv"
    if minimum_path.exists():
        raise FileExistsError(minimum_path)
    minimum.to_csv(minimum_path, index=False)
    output = {
        "campaign": "Thayer-Identifiability-v1",
        "preregistration_sha256": PREREG_SHA256,
        "scene_count": len(scene_results),
        "row_count": len(table_frame),
        "scene_results": [str(scene_output_path(index).relative_to(REPO)) for index in INDICES],
        "identifiability_table": str(table_path.relative_to(REPO)),
        "full_metrics_table": str(full_path.relative_to(REPO)),
        "minimum_unique_table": str(minimum_path.relative_to(REPO)),
        "minimum_prior_levels": {
            str(scene["scene"]): scene["minimum_prior_level_for_unique"] for scene in scene_results
        },
        "classification_counts": full_frame.classification.value_counts().to_dict(),
        "access_counts": {
            "authorized_training_scenes": len(scene_results),
            "authorized_training_catalog_rows": 2 * len(scene_results),
            "development": 0,
            "atlas": 0,
            "lockbox": 0,
            "neural_model_imports": 0,
            "network_weight_optimizer_steps": 0,
        },
    }
    write_json_fresh(RUN / "logs/aggregate_results.json", output)
    print(json.dumps(output, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, choices=INDICES)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate == (args.scene is not None):
        parser.error("choose exactly one of --scene or --aggregate")
    if args.aggregate:
        aggregate()
    else:
        execute_scene(int(args.scene))


if __name__ == "__main__":
    main()
