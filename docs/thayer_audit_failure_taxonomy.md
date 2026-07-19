# Thayer-Audit failure taxonomy

Every label is positive, negative, or not applicable; labels are not collapsed
into an unexplained binary target.

- `QUERY_NULL`: no valid source matches the prompt.
- `QUERY_AMBIGUOUS`: more than one source matches within the frozen ambiguity
  margin.
- `SOURCE_CONFUSION`: the candidate is closer to an unrequested scene source
  than to the requested truth.
- `CATASTROPHIC_IMAGE`: requested-source image distance exceeds 0.25.
- `CATASTROPHIC_FLUX`: any g/r/z relative flux error exceeds 0.20.
- `CATASTROPHIC_CENTROID`: centroid error exceeds 0.5 mean-PSF FWHM when both
  centroids are defined.
- `COLOR_UNSAFE`: either defined g-r or r-z error exceeds 0.20 mag.
- `SHAPE_UNSAFE`: a frozen valid shape metric exceeds its limit; not applicable
  in the current feasibility branch because the shape gate was not validated.
- `ATLAS_NON_IDENTIFIABLE`: the observation belongs to a validated Atlas pair.
- `SAFE_CANDIDATE`: no applicable candidate failure is positive.

Truth may form training/evaluation labels but is never a deployable auditor
input. In the frozen Atlas diagnostic, query-null and query-ambiguous are not
applicable because prompts are valid. Among 150 noisy model decompositions,
all 150 were catastrophic in image and flux, none were safe, and 104 were
closer to the unrequested source. These are stress-benchmark labels, not
catalog-policy rates.
