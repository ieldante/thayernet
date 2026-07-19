# Thayer-PU eligibility v1

The authoritative run is
`outputs/runs/thayer_pu_eligibility_v1_20260714_213113/`; preregistration
SHA-256 is
`6f5cd5de57e7810aab947c9c59e955bb09215abb5d32251dff663cbf753d578c`.
The authoritative outcome is **THAYER_PU_DEPLOYMENT_INELIGIBLE**.

Thayer-Audit v0 was POST-untestable because Condition C supplied no safe
examples. This campaign replayed all 12,493 existing Condition-C labels with
the unchanged safety implementation: 12,493 unsafe, zero safe, and zero
substantive label or metric mismatches. Two initially flagged values were
`1.8189894035458565e-12` CSV decimal round-trip differences in large baseline
distances; every component and final label matched.

Thayer-PU checkpoint
`outputs/runs/thayer_probabilistic_unet_20260712_163340/checkpoints/thayer_pu_best.pth`
at SHA-256
`c1d17a3f67962cce2fec03d6b15da5f2e330ee97b31c270a7ff019a1373a557e`
was evaluated unchanged. The sole frozen deployment rule averaged 16
truth-free prior-predictive requested-source samples, using seeds
2026077600–2026077615. It used no truth, candidate selection, development
outcomes, or Atlas outcomes.

Promptability passed on the source-group-safe preflight: majority-of-16 and
individual requested identity were both 1.0; per-band identity was
1.0/1.0/0.9505. Repeated batch-8 inference and batch-4 versus batch-8 hashes
were exact. Single-scene inference changed every candidate and deployed hash
for all 24 preregistered scenes. Maximum deployed absolute differences were
`0.0005760`–`0.0007668` detected electrons. The frozen contract required exact
single-scene/batched consistency, so the run stopped before full inference.

No Thayer-PU safety labels, prevalence estimates, family comparison,
label-support bootstrap, combined inventory, or auditor fit was produced.
The 3,998/793/2,800 leak-free training/validation/calibration source manifests
remain provenance evidence only. Thayer-PU is not an eligible second frozen
family, and Thayer-Audit v1 is not authorized.

Exactly one next experiment is recommended: **Thayer-Audit Family-D v0 — One
New Physically Compliant Frozen Family Eligibility Audit**. It must freeze one
new family and one truth-free deployment rule before labels, require exact
replay and physical nonnegativity, and preserve the same OOF, safety, and
privacy contracts.

## Fixed-batch repair and Family-E disposition

The append-only Batch-R1 repair at
`outputs/runs/thayer_pu_batch_r1_20260714_224244/` used fixed eight-row MPS
geometry and passed replay without changing the checkpoint. Complete outputs
still collapsed to 0 safe / 7,591 unsafe.

The subsequent Family-E v0 run stopped before model construction: raw
zero-background observations are signed and cannot equal a sum of three
nonnegative simplex allocations. No Family-E labels or auditor were produced;
Thayer-Audit v1 remains unauthorized.
