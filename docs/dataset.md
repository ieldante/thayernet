# Dataset Notes

## Dataset

The project uses Galaxy10 DECaLS, a public galaxy morphology dataset distributed as an HDF5 file. The file is not included in this repository and must be downloaded separately from the official dataset source.

Expected local path:

```text
data/Galaxy10_DECals.h5
```

## Expected HDF5 Contents

The loader expects:

- `images`: RGB galaxy cutouts with shape similar to `(N, 256, 256, 3)`.
- `ans`: integer morphology labels.

If present, the loader also returns metadata arrays:

- `ra`
- `dec`
- `redshift`
- `pxscale`

The labels are not used as supervision for deblending, but they may be useful for later analysis by morphology class.

## Data Handling Policy

- Do not commit or publish `Galaxy10_DECals.h5`.
- Keep the dataset under `data/` locally.
- Keep `data/.gitkeep` so the directory exists in a fresh clone.
- Regenerate synthetic blends after code changes instead of committing cached blend arrays.

## Memory Notes

The full image array is large. For exploratory notebook work, use small subsets first to validate the pipeline. Larger runs should use batching and may require more memory-aware data loading in future iterations.

## Preprocessing

Images are normalized from integer pixel values to `float32` values in `[0, 1]`.
The historical pipeline split HDF5 row indices before blending, but duplicate
source rows still crossed train/validation/test. The grouped development
protocol instead keeps exact-pixel and exact-coordinate groups wholly within
one partition and records source/group IDs for both blend roles. This fixes the
demonstrated exact leakage for grouped development work; it does not provide an
untouched final-paper partition or prove exhaustive near-duplicate identity.
