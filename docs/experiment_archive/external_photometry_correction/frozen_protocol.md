# Frozen convergence-correction protocol

Frozen before any correction-campaign scientific optimizer call.

- Campaign: `Thayer-External-Photometry-Convergence-Correction-v0`.
- Predecessor: exact append-only `Thayer-External-Photometry-Preflight-v0` run `20260718_154852`, required outcome `PREFLIGHT_OPTIMIZATION_LIMITED` and required next experiment `Thayer-External-Photometry-Convergence-Correction-v0`.
- Only authorized scientific-execution change: maximum function evaluations per start, `150 -> 500`.
- Scope unchanged: Scenes 0 and 6 only; Level 5 / bulge+disk only; `TOTAL_SOURCE_PHOTOMETRY` and `PER_BAND_SOURCE_PHOTOMETRY` only.
- Exactly four deterministic starts per fit. Physical start vectors and initialization hashes must match the preflight byte-for-byte. The implementation is directly bounded in physical coordinates; there is no unconstrained latent initialization vector.
- Exact frozen noisy measurements are loaded from the preflight table with SHA-256 `7767fd176b1b062959d58ed82f075563e263474728ee669f3d85d5fbd8a65e8d`. Measurements are not regenerated. Relative sigma remains 5% and photometry remains an explicit Gaussian likelihood term.
- Frozen Model-9 renderer, parameterization, morphology support, PSF, observation, noise convention, objective, optimizer (`scipy.optimize.least_squares`, bounded TRF, `x_scale=jac`), `1e-10` tolerances, gradient diagnostic, endpoint support/acceptance, image clustering, rank/nullity, uniqueness diameters, boundary rules, and replay procedure remain unchanged.
- Every endpoint is retained, including optimizer-declared failures and starts reaching 500 evaluations. Fits run sequentially; no favorability-based retry or early stop is permitted.
- The best endpoint is replayed by two exact render/residual/Jacobian evaluations and all three hashes must match.
- Fit labels preserve the preflight promise logic: `STRONG_PROMISE`, `MODERATE_PROMISE`, and `NO_CLEAR_GAIN` map to `CONVERGED_STRONG_PROMISE`, `CONVERGED_MODERATE_PROMISE`, and `CONVERGED_NO_CLEAR_GAIN`; preflight optimization-limitation logic maps to `STILL_OPTIMIZATION_LIMITED`. A strong label additionally requires the frozen `1e-5` gradient gate. Contract or integrity failure maps to `INVALID`.
- Campaign labels and the exactly-one-next-experiment rule are those specified for this correction campaign.
- No isolated-source image, morphology truth, mask, truth initialization, catalog truth flux, protected development data, Atlas tensor, or lockbox may enter inference. No measurement generation or neural training is authorized.

Pre-execution estimate from the frozen preflight: 562.870292 seconds for 1461 evaluations. If only the five previously capped starts use the full extension, the linear estimate is 1237.082 seconds (20.62 minutes). The all-cap linear bound is 1876.234 seconds (31.27 minutes). No 30-minute wall-clock limit is imposed.
