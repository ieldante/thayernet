# Thayer-External-Photometry-Stratification-Convergence-Correction-v0 frozen protocol

Frozen before any correction-campaign scientific optimizer call.

## Scientific scope

- Predecessor: the completed append-only `Thayer-External-Photometry-Scene-Stratification-v0` run `20260719_011606`.
- Scenes: exactly 5 and 18.
- Structural family: exactly Model-9 Level 5 / bulge+disk.
- Condition: exactly `TOTAL_SOURCE_PHOTOMETRY`.
- Starts: exactly the same four deterministic direct-bounded physical starts per scene as the predecessor.
- Measurements: exactly the predecessor's noisy total-source-photometry measurements and 5% uncertainties, loaded from the predecessor table with SHA-256 `7fdca1ab99c9ca9e7f6a186043a86c313ad3e3ccc29e072c5cd549ece25f3c95`; they are not regenerated or perturbed and remain an explicit Gaussian likelihood term.
- Only authorized scientific-execution change: maximum function evaluations per start, `500 -> 2000`.

No per-band photometry, Level 4 family, other scene, alternate uncertainty, PSF change, new start, endpoint initialization, truth initialization, neural training, or audit training is authorized.

## Frozen solver and diagnostics

The predecessor's renderer, direct bounded physical parameterization, source family, observation likelihood, photometry likelihood, objective, measurement uncertainty, PSF, source coordinates, morphology support, bounded-TRF optimizer (`scipy.optimize.least_squares`, `method="trf"`, `x_scale="jac"`), `ftol/xtol/gtol=1e-10`, gradient gate, endpoint support and acceptance, endpoint clustering, rank/nullity rules, condition computation, scientific diameters, replay, and classification logic remain unchanged.

All four endpoints are retained for each scene, including failures and 2000-evaluation endpoints. Fits run sequentially with no favorability-based retry. The residual wrapper records every objective evaluation without changing the returned residual or Jacobian. The best finite endpoint is replayed twice with exact requested-image, companion-image, residual, and Jacobian hashes.

The frozen numerical perturbation diagnostic uses the predecessor Model-9 rule: add `1e-8` times the dimensionless parameter scale in the deterministic sine direction, clip to frozen bounds, and require finite residual/Jacobian, unchanged rank/nullity, and singular-spectrum relative change no larger than `1e-3`. This diagnostic does not change the frozen scientific classifier.

## Interpretation and correction analysis

A scene is interpretable only when the unchanged external-photometry scientific classifier is not `OPTIMIZATION_UNRESOLVED`, `NUMERICALLY_UNSTABLE`, `OUT_OF_SUPPORT`, or `INVALID_CONTRACT`. For an interpretable scene, the frozen predecessor helpfulness rule is applied relative to authoritative P2: fewer endpoint classes, or no diameter worsening plus at least 50% reduction of every nonzero P2 scientific diameter. Otherwise the scene is not helpful. An uninterpretable scene is `STILL_OPTIMIZATION_UNRESOLVED`; contract or integrity failure is `INVALID`.

If at least one scene resolves, the six predecessor labels remain frozen and only newly interpretable Scene 5/18 labels are appended in separately named corrected outputs. Exact conditional label-permutation p-values, Benjamini-Hochberg q-values, exact Clopper-Pearson confidence intervals, and leave-one-out balanced accuracy use the predecessor algorithms. The six-scene artifacts are never modified. No predictor is claimed as validated from at most eight scenes.

## Integrity

The authorization gate must verify predecessor completion and exact recommendation; Scenes 5 and 18 as the sole unresolved primary cases; all eight prior endpoints at the 500-evaluation ceiling; predecessor artifact and measurement hashes; exact regeneration of start IDs, vectors, and initialization hashes; Model-9 readiness; relevant tests and standalone validation; 600/600 historical checkpoints; unchanged README and HEAD; empty Git index; no stale optimizer; and zero protected-data access. Any failure stops scientific execution fail-closed.

Protected development data, Atlas tensors, lockbox data, isolated-source arrays, catalog truth, masks, truth morphology, and prior endpoints are prohibited solver inputs. Outputs are append-only in this fresh run directory. Nothing is staged or committed.
