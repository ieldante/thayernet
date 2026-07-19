# Thayer-Model-9-Preparation-v0 final report

## Status

**MODEL_9_FOUNDATION_READY**

The engineering foundation is complete and the separately preregistered
**Thayer-Flux-Free-Identifiability-v0** campaign may now run. This preparation
did not execute that experiment, load any of its eight observations, fit a
scientific scene, compute a scientific uniqueness result, or train a neural
network.

## Exact implementation findings

Six fresh source/test/validation files were added without modifying existing
files:

- `src/model9_structured.py`: differentiable Level-4 Sérsic and Level-5
  bulge+disk rendering; free nonnegative unbounded-above g/r/z source fluxes;
  grouped PSF convolution; signed residual; Gaussian likelihood; parameter
  support and scaling; symmetry quotient; canonical hashing; solver-input
  provenance and oracle rejection.
- `src/model9_galsim_adapter.py`: deterministic conversion of caller-supplied
  known GalSim PSFs to normalized 31x31 g/r/z kernels with fail-closed ringing
  tolerances.
- `src/model9_optimizer.py`: deterministic 16-start schedule, bounded
  trust-region least squares, autograd Jacobian, SVD rank/null analysis,
  Gauss-Newton Hessian, condition diagnostics, endpoint clustering and
  diameters, boundary flags, prompt identity, replay, chi-square support gate,
  and the frozen eight-label classifier.
- `src/model9_synthetic.py`: separated identifiable, coincident ambiguous,
  bulge+disk, pure-component boundary, and PSF fixtures.
- `tests/test_model9_foundation.py`: 25 exhaustive synthetic preparation tests.
- `scripts/validate_thayer_model9_foundation.py`: standalone synthetic-only
  machine-readable readiness validator.

The inference object has no field for isolated source images, per-source truth
fluxes, true source parameters, morphology labels, catalog morphology, or
truth initialization. The primary flux support is `[0, infinity)` for each
source and band. Observation/noise total information is used only to create
diverse starts and dimensionless diagnostic scales; it is not a flux
constraint, truth surrogate, or penalty.

## Protocol freeze

The single primary next-campaign protocol is
`preregistration/draft_flux_free_protocol.md`, SHA-256
`5b37499d0ea957ddb36b3737a5c24ae4aa489f5fd066b3012dafbb475157695b`.
Its machine-readable companion has SHA-256
`8235923adabe28407fcceabc883448c4afd9006a5e00b633170430a191f6e692`.

The freeze fixes:

- exact original Level-4/5 morphology bounds and prompt-centered identities;
- direct nonnegative source fluxes with no upper flux prior;
- 4x differentiable pixel integration and 31x31 normalized known PSFs;
- signed residual `O-S_req-S_comp` and one Gaussian likelihood objective;
- 16 starts, seed `2026071519`, 500 evaluations, and `1e-10` optimizer
  tolerances for both families;
- complete symmetry quotient and boundary-collapse handling;
- float64 rank, null basis, condition, Hessian, gradient, endpoint, flux, and
  morphology diagnostics;
- a 0.99 chi-square support gate, `1e6` condition ceiling, `1e-3` uniqueness
  diameters, `1e-5` gradient ceiling, and all eight mutually exclusive
  classifications.

No alternate objective or post-result parameterization is authorized as the
primary analysis.

## Verification evidence

All 25 Model-9 tests pass. The combined authorized compatibility suite has
40/40 passing tests across Model 9, canonical tensor hashing, the authoritative
Family-E signed residual, and existing PSF conditioning. Bytecode compilation
passes. The authoritative standalone validation is
`engineering_validation/synthetic_validation_r2.json`, SHA-256
`a353f5a7e6609f7425bfab5c58ab446bc1e405f445c3e32376417e75b0b249da`.

The independent validation shows:

- GalSim renderer relative L2 error `0.0011263` for n=1 and `0.0332735` for
  n=4, below frozen `0.005` and `0.05` thresholds;
- explicit g/r/z PSF normalization error `0`;
- physical source flux round-trip error at most `2.84e-14` and signed closure
  error at most `8.88e-16`;
- finite, nonnegative requested and companion layers for all fixtures;
- autograd/finite-difference relative errors `2.82e-10` for Sérsic and
  `1.77e-10` for bulge+disk;
- nontruth-initialized exact synthetic recovery in 7 evaluations, chi-square
  `3.05e-28`, gradient norm `3.29e-12`, and maximum parameter error
  `2.84e-14`;
- exact replay SHA-256 equality across two complete reruns;
- separated flux rank/null `6/0`, full structured rank/null `14/0`, synthetic
  `UNIQUE` classification;
- coincident identical-source flux rank/null `3/3`, two exact endpoint
  classes, infinite condition, synthetic `NON_IDENTIFIABLE` classification;
- exact zero-error pure-disk and pure-bulge boundary equivalences after gauge
  handling;
- passing oracle negative controls that reject `true_per_source_flux`.

These are engineering fixtures only. They reveal no result for the frozen
eight scientific scenes.

## Integrity and access

- frozen scientific observation arrays accessed: 0;
- isolated source arrays accessed: 0;
- development accessed: 0;
- Atlas arrays accessed: 0;
- lockbox accessed: 0;
- neural models imported or trained: 0;
- scientific scene optimizer starts: 0.

The authoritative 600-entry historical checkpoint inventory was independently
rehashed: 600 matched, 0 differed, 0 were missing, across 210,297,303 bytes.
README SHA-256 remains
`67f66f351f8d1de56f760608b4dbe663e13590ae856012b6b7a0eeb2ec0116a1`.
The Git index is empty and no commit was created. The 376 pre-existing
workspace entries remain user-owned; the campaign added exactly six fresh
source/test files plus this ignored append-only run directory.

The requested `Thayer-Project-Synthesis-v1` artifact was not present under
that title. The foundation reconciled the superseding Recoverability and
Identifiability final reports, the frozen Identifiability prior specification,
the signed-residual preflight, current status, roadmap, and the explicit
campaign contract. No missing synthesis text was used to invent a numerical
choice.

## What remains

No engineering prerequisite remains for the controlled eight-scene
identifiability experiment. The future campaign must still verify all source
and protocol hashes before scene access, record each observation identifier
and hash, rerun the preparation tests, persist every start and endpoint, and
stop invalid if any provenance or replay check changes. Actual scientific
support, uniqueness, and the survival of the previous 7/8 conditional result
remain completely unknown.

PriorNet and POST remain unauthorized. A direct structural reconstruction
campaign is not authorized until flux-free identifiability has produced its
own valid outcome.

## Exactly one recommended next experiment

Run **Thayer-Flux-Free-Identifiability-v0** under the frozen protocol and source
hashes above, on exactly the eight already frozen observations, with no oracle
per-source fluxes and no neural training.
