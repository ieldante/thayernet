# Scientific and Postprocessing Isolation

The future D3 execution path and visualization path are separate launchers.

The scientific readiness launcher imports NumPy, PyTorch, and the exact model,
mapping, and evaluator modules needed before a D3 tensor load. Its closed
strict-phase graph has no Matplotlib edge. It performs no plotting and writes
no package cache. A regression test also proves the historical D3R edge from
its runner to a module with a top-level Matplotlib import.

The postprocessing launcher runs in a separate disposable sandbox. It may
import Matplotlib only after scientific outputs have been frozen. Its input is
restricted to explicitly produced new-run tables or artifacts, and its output
is restricted to postprocessing scratch. It cannot read a scene, target,
cached feature, endpoint tensor, protected partition, or historical scientific
artifact directly, and it cannot modify scientific outputs.

Thayer-D3B exercised the postprocessor with synthetic values only. The probe
produced one synthetic status figure, allowed zero scientific-artifact reads,
and confined its two lifecycle operations to its disposable runtime. It
mutated no scientific artifact. Atlas, development, and lockbox remained
untouched.
