# D3 v4.1 R1 production checkpoint prewarm

The R1 adapter derives the exact 11-key `thayer-d3-checkpoint-v4` structure
from the frozen v4 writer and reader contexts. Its deterministic synthetic
payload contains two production-compatible expert state dictionaries, an
exercised AdamW state, metrics, physical and penultimate tensors, and frozen
identifier fields. It uses `map_location=cpu`, `weights_only=True`, and the
unchanged exclusive-write serialization behavior.

Two guarded cold processes and one warm process recorded one writer call, at
least two reader calls, the complete serialization-module set before strict
mode, zero strict new imports, and zero strict external `.pyc` reads. No
scientific checkpoint or model value was used.
