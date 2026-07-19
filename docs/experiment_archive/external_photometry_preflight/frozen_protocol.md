# Frozen preflight protocol

Frozen before external-measurement generation or scientific fitting.

- Campaign: `Thayer-External-Photometry-Preflight-v0`
- Scope: Scene 0 and Scene 6 only; Level 5 / bulge+disk only.
- Historical controls: authoritative S1 `FLUX_FREE_SINGLE` and P2 `PSF_DIVERSE`, comparison only; neither is rerun.
- New conditions: `TOTAL_SOURCE_PHOTOMETRY` and `PER_BAND_SOURCE_PHOTOMETRY` on the unchanged single observation.
- Renderer, parameterization, morphology support, PSF, observation noise convention, optimizer tolerances, symmetry quotient, endpoint clustering, and diameter definitions: frozen Model-9 implementation.
- External measurements: CatSim catalog g/r/z AB photometry converted with the same documented LSST `mag2counts` calibration used by BTK. No isolated image is read or generated.
- Total-photometry combination: `1*g + 1*r + 1*z` detected-electron flux, one scalar per source.
- Per-band photometry: separate g, r, z detected-electron fluxes per source.
- Measurement model: independent Gaussian, sigma = 0.05 times the latent catalog flux (or weighted total); deterministic synthetic noise seed `2026071805`. Only noisy measured values and declared sigma enter fitting.
- Fits: exactly 4 deterministic Model-9 starts per new scene/condition; the same four starts are reused across photometry conditions; maximum 150 function evaluations per start; CPU float64.
- Objective residual vector: frozen whitened single-observation residual concatenated with external-photometry standardized residuals. Observation and photometry log likelihoods are reported separately.
- Acceptable endpoint classes: finite optimizer endpoints that pass the frozen observation chi-square support gate, then the frozen total-objective tolerance and image-space clustering. All four endpoints are retained regardless of status.
- Local rank/nullity/condition: symmetry-corrected SVD of the combined observation-plus-photometry residual Jacobian using frozen parameter scaling.
- Replay: render, combined residual, and combined Jacobian are evaluated twice at the best endpoint and must hash exactly.
- No isolated-source image, mask, morphology truth/label, truth initialization, protected development, Atlas tensor, or lockbox may be accessed. Catalog truth photometry is confined to the deterministic measurement generator and discarded before fitting.

Preflight promise rules are literal. `STRONG_PROMISE` requires one acceptable class and every scientific diameter to be strictly smaller than both S1 and P2. `MODERATE_PROMISE` requires a lower endpoint-class count than P2 or at least a 50% reduction in every nonzero P2 scientific diameter without worsening any diameter. `NO_CLEAR_GAIN` applies otherwise. A fit is `OPTIMIZATION_LIMITED` if it has no optimizer-declared successful endpoint, no observation-support endpoint, a failed replay, or a nonfinite diagnostic. Any information-contract failure is `INVALID`.
