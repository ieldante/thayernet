"""Deterministic, identity-preserving BTK engineering scene utilities.

This module is deliberately independent of the historical RGB and FITS
blenders.  It uses BTK's public ``CatsimCatalog``, ``SamplingFunction``,
``CatsimGenerator``, and ``get_surveys`` interfaces.  Catalog identity is kept
outside the renderer as the immutable tuple (catalog SHA-256, catalog row,
``galtileid``); BTK is not relied upon to invent or preserve an identifier.

BTK 1.0.9's verified observation contract is:

* ``isolated_images`` are noiseless, PSF-convolved electron-count images;
* ``add_noise='none'`` returns their summed scene;
* ``add_noise='all'`` adds source Poisson noise and one zero-mean sky Poisson
  realization to that summed scene, never independently to isolated images.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy.table import Table
from btk.catalog import CatsimCatalog
from btk.draw_blends import CatsimGenerator
from btk.sampling_functions import SamplingFunction
from btk.survey import get_surveys

BAND_ORDER = ("g", "r", "z")
SURVEY_NAME = "LSST"
STAMP_SIZE_ARCSEC = 12.0
PIXEL_SCALE_ARCSEC = 0.2
IMAGE_UNITS = "detected electrons per pixel"
REQUIRED_COLUMNS = {
    "galtileid",
    "ra",
    "dec",
    "redshift",
    "fluxnorm_bulge",
    "fluxnorm_disk",
    "fluxnorm_agn",
    "a_b",
    "a_d",
    "b_b",
    "b_d",
    "pa_bulge",
    "pa_disk",
    "g_ab",
    "r_ab",
    "z_ab",
}


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def persistent_source_id(catalog_hash: str, catalog_row: int, galtileid: int) -> str:
    """Build an immutable external CatSim source identifier."""

    return f"catsim:{catalog_hash}:row:{catalog_row}:galtileid:{galtileid}"


def load_catsim_catalog(path: Path) -> tuple[CatsimCatalog, str]:
    """Load and validate the versioned engineering CatSim catalog."""

    path = path.resolve()
    digest = file_sha256(path)
    raw = Table.read(path)
    missing = sorted(REQUIRED_COLUMNS.difference(raw.colnames))
    if missing:
        raise ValueError(f"CatSim catalog lacks required columns: {missing}")
    if len(raw) == 0:
        raise ValueError("CatSim catalog is empty")
    for name in REQUIRED_COLUMNS:
        values = np.asarray(raw[name])
        if np.issubdtype(values.dtype, np.number) and not np.all(np.isfinite(values)):
            raise ValueError(f"CatSim column {name!r} contains non-finite values")
    raw["catalog_row"] = np.arange(len(raw), dtype=np.int64)
    raw["engineering_source_id"] = np.asarray(
        [
            persistent_source_id(digest, i, int(galtileid))
            for i, galtileid in enumerate(raw["galtileid"])
        ]
    )
    return CatsimCatalog(raw), digest


class FixedSceneSampling(SamplingFunction):
    """Return declared catalog rows at declared arcsecond offsets.

    This public ``SamplingFunction`` implementation removes all hidden source
    or position sampling from rendering.  ``ra`` and ``dec`` are BTK tangent
    plane offsets in arcseconds, as established from BTK 1.0.9 source and WCS
    probes; they are not the original catalog sky coordinates.
    """

    def __init__(
        self,
        catalog_rows: Sequence[int],
        positions_arcsec: Sequence[tuple[float, float]],
        stamp_size: float = STAMP_SIZE_ARCSEC,
    ) -> None:
        if len(catalog_rows) != len(positions_arcsec):
            raise ValueError("catalog_rows and positions_arcsec lengths differ")
        if not catalog_rows:
            raise ValueError("A fixed scene must contain at least one source")
        super().__init__(
            stamp_size=int(stamp_size),
            min_number=len(catalog_rows),
            max_number=len(catalog_rows),
            seed=0,
        )
        self.stamp_size = float(stamp_size)
        self.catalog_rows = tuple(int(value) for value in catalog_rows)
        self.positions_arcsec = tuple((float(x), float(y)) for x, y in positions_arcsec)
        if max(max(abs(x), abs(y)) for x, y in self.positions_arcsec) >= stamp_size / 2:
            raise ValueError("A requested source coordinate lies outside the stamp")

    def __call__(self, table: Table) -> Table:
        output = table[list(self.catalog_rows)].copy()
        output["ra"] = np.asarray([position[0] for position in self.positions_arcsec])
        output["dec"] = np.asarray([position[1] for position in self.positions_arcsec])
        return output


@dataclass(frozen=True)
class SceneSpec:
    """Complete deterministic scene request."""

    scene_id: str
    catalog_rows: tuple[int, ...]
    positions_arcsec: tuple[tuple[float, float], ...]
    source_selection_seed: int
    position_seed: int
    noise_seed: int


@dataclass
class SceneRender:
    """BTK output normalized to the explicit g,r,z campaign contract."""

    blend: np.ndarray
    isolated: np.ndarray
    psf: np.ndarray
    catalog: Table
    bands: tuple[str, ...]
    full_survey_bands: tuple[str, ...]
    pixel_scale_arcsec: float


def validated_lsst_survey():
    """Return the documented LSST survey after strict band/scale checks."""

    survey = get_surveys(SURVEY_NAME)
    absent = [band for band in BAND_ORDER if band not in survey.available_filters]
    if absent:
        raise RuntimeError(f"{SURVEY_NAME} lacks requested bands: {absent}")
    pixel_scale = float(survey.pixel_scale.to_value("arcsec"))
    if not np.isclose(pixel_scale, PIXEL_SCALE_ARCSEC, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Unexpected LSST pixel scale {pixel_scale}")
    return survey


def render_fixed_scene(
    catalog: CatsimCatalog,
    spec: SceneSpec,
    *,
    add_noise: str,
) -> SceneRender:
    """Render one fixed scene through the verified public BTK API."""

    if add_noise not in {"none", "all"}:
        raise ValueError("Engineering scenes allow only add_noise='none' or 'all'")
    survey = validated_lsst_survey()
    sampler = FixedSceneSampling(spec.catalog_rows, spec.positions_arcsec)
    generator = CatsimGenerator(
        catalog,
        sampler,
        survey,
        batch_size=1,
        njobs=1,
        verbose=False,
        use_bar=False,
        add_noise=add_noise,
        seed=spec.noise_seed,
        apply_shear=False,
        augment_data=False,
    )
    batch = next(generator)
    full_bands = tuple(batch.survey.available_filters)
    band_indices = [full_bands.index(band) for band in BAND_ORDER]
    blend = np.asarray(batch.blend_images[0, band_indices], dtype=np.float64)
    isolated = np.asarray(
        batch.isolated_images[0, : len(spec.catalog_rows)][:, band_indices], dtype=np.float64
    )
    image_size = blend.shape[-1]
    scale = float(batch.survey.pixel_scale.to_value("arcsec"))
    psf = np.asarray(
        [
            batch.psf[index]
            .drawImage(nx=image_size, ny=image_size, scale=scale)
            .array.astype(np.float64)
            for index in band_indices
        ]
    )
    if blend.shape != (3, image_size, image_size):
        raise RuntimeError(f"Unknown blend shape/order: {blend.shape}")
    if isolated.shape != (len(spec.catalog_rows), 3, image_size, image_size):
        raise RuntimeError(f"Unknown isolated shape/order: {isolated.shape}")
    if not np.all(np.isfinite(blend)) or not np.all(np.isfinite(isolated)):
        raise RuntimeError("Non-finite renderer output")
    rendered_catalog = batch.catalog_list[0]
    if list(np.asarray(rendered_catalog["catalog_row"], dtype=int)) != list(spec.catalog_rows):
        raise RuntimeError("BTK output catalog rows do not match the fixed request")
    return SceneRender(
        blend=blend,
        isolated=isolated,
        psf=psf,
        catalog=rendered_catalog,
        bands=BAND_ORDER,
        full_survey_bands=full_bands,
        pixel_scale_arcsec=scale,
    )


def eligible_engineering_rows(table: Table) -> np.ndarray:
    """Return bright, visible CatSim rows suitable for small engineering scenes."""

    component_flux = (
        np.asarray(table["fluxnorm_bulge"], dtype=float)
        + np.asarray(table["fluxnorm_disk"], dtype=float)
        + np.asarray(table["fluxnorm_agn"], dtype=float)
    )
    size = np.maximum(np.asarray(table["a_b"], dtype=float), np.asarray(table["a_d"], dtype=float))
    condition = (
        (np.asarray(table["i_ab"], dtype=float) >= 20.0)
        & (np.asarray(table["i_ab"], dtype=float) <= 24.5)
        & (component_flux > 0)
        & (size >= 0.15)
        & (size <= 1.5)
    )
    return np.flatnonzero(condition)


def build_scene_specs(table: Table, source_selection_seed: int = 2026071101) -> list[SceneSpec]:
    """Select 60 disjoint engineering-only sources and construct 40 scenes."""

    eligible = eligible_engineering_rows(table)
    if len(eligible) < 60:
        raise RuntimeError(f"Only {len(eligible)} eligible engineering sources")
    selection_rng = np.random.default_rng(source_selection_seed)
    selected = selection_rng.choice(eligible, size=60, replace=False)
    specs: list[SceneSpec] = []
    for index in range(20):
        position_seed = 2026071200 + index
        rng = np.random.default_rng(position_seed)
        position = (float(rng.uniform(-0.25, 0.25)), float(rng.uniform(-0.25, 0.25)))
        specs.append(
            SceneSpec(
                scene_id=f"single_{index + 1:03d}",
                catalog_rows=(int(selected[index]),),
                positions_arcsec=(position,),
                source_selection_seed=source_selection_seed,
                position_seed=position_seed,
                noise_seed=2026071300 + index,
            )
        )
    for index in range(20):
        position_seed = 2026072200 + index
        rng = np.random.default_rng(position_seed)
        separation = float(rng.uniform(1.2, 2.4))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        dx = 0.5 * separation * np.cos(angle)
        dy = 0.5 * separation * np.sin(angle)
        jitter = rng.uniform(-0.12, 0.12, size=2)
        positions = (
            (float(jitter[0] - dx), float(jitter[1] - dy)),
            (float(jitter[0] + dx), float(jitter[1] + dy)),
        )
        rows = (int(selected[20 + 2 * index]), int(selected[21 + 2 * index]))
        specs.append(
            SceneSpec(
                scene_id=f"double_{index + 1:03d}",
                catalog_rows=rows,
                positions_arcsec=positions,
                source_selection_seed=source_selection_seed,
                position_seed=position_seed,
                noise_seed=2026072300 + index,
            )
        )
    return specs


def gaussian_prompt(
    shape: tuple[int, int], x_pixel: float, y_pixel: float, sigma_pixel: float = 2.0
) -> np.ndarray:
    """Create a unit-peak subpixel Gaussian requested-source coordinate channel."""

    if sigma_pixel <= 0:
        raise ValueError("Prompt sigma must be positive")
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    prompt = np.exp(-0.5 * ((xx - x_pixel) ** 2 + (yy - y_pixel) ** 2) / sigma_pixel**2)
    return prompt.astype(np.float64)
