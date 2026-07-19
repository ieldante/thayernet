# Direct scientific-feasibility projection

Thayer-FP is the projection and micro-feasibility campaign in
`outputs/runs/thayer_feasibility_projection_20260712_234216/`. Its
preregistration SHA-256 is
`c826734eab4d299b875aa7d69529816e6ca1db1cdefa63400deea334bc29e4d8` and
predates every per-scene array load, projection, and neural optimizer step.

All Thayer-ME, Thayer-SA, and Thayer-OC baselines reproduced. The
actual-objective HVP status remains unresolved; Thayer-FP performed no new HVP
or condition-number analysis. Every exact approved target passed the unchanged
image, flux, color, centroid, source-layer, assignment, canonical-hash, and
truth-coverage checks.

The guaranteed truth homotopy projected all 256 expert/prompt pairings into the
strict 0.95 training interior. Median interior alpha was 0.999979 and median
normalized correction was 0.946369, showing that the persisted compromises
had to move almost completely to their approved targets: the frozen P0
representatives were nearly exact truth. Flux-z was the most
frequent limiting constraint. Seven scientific component paths were
nonmonotone, but binary feasibility was monotone for every pairing.

P1 reduced median correction to 0.885394 and passed the unchanged scientific
threshold of 1.0, but three pairings reached 0.950001 and therefore missed the
separate preregistered training-interior ceiling. P1 was not frozen. The final
global method is P0. Its projected sets achieved 100% ordinary, own,
alternate, and both-mode coverage, 100% ordinary and ambiguous forward
consistency, and stable deterministic hashes without changing any scientific
threshold.

Projection succeeded, but the unchanged Thayer-ME failed to learn the targets.
After 400 MPS-only epochs, all four scientific coverage categories remained
zero, ordinary expert diameter was 3.564, and 43.6% of output pixels were
negative. Prompt swap remained 0.984 and forward consistency remained
0.969/1.000. This directly implicates decoder capacity, encoder conditioning,
or the neural output parameterization on the microset; it does not implicate
the existence or reachability of the frozen scientific region.

The follow-up Thayer-CL preflight reproduced these results but found that the
frozen contracts do not identify one unique nonnegative neural output mapping.
It stopped before constructing a capacity condition. The one authorized next
experiment is now a separately preregistered output-parameterization campaign
at fixed L0 capacity. Atlas, development, and lockbox data remain prohibited.

The completed Thayer-OP follow-up confirms that ReLU, square, and absolute
value can each represent every unchanged P0 target and enforce nonnegative
physical outputs. Under otherwise identical L0 training, however, every
mapping finished with zero ordinary coverage and zero ambiguous both-mode
coverage. The P0 targets remain unchanged and feasible; the unresolved blocker
is their one-scene memorization through the frozen neural decoder path. No
eight-scene or capacity condition was run.
