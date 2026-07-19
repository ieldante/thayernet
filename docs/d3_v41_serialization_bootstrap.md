# D3 v4.1 serialization bootstrap

The v4 failure included lazy strict-phase reads of
`torch/utils/serialization/__init__.cpython-39.pyc` and
`torch/utils/serialization/config.cpython-39.pyc`. V4.1 explicitly imports
both proven modules before strict mode and exercises one tiny synthetic
`torch.save` plus `torch.load(..., weights_only=True)` inside fresh runtime
scratch. The synthetic object contains no scientific tensor or model
parameter.

Candidate 001 stopped before bridge creation on a Correction-B report-helper
reference and remains preserved. Candidate 002 used fresh paths, passed the
prewarm, removed its scratch checkpoint and directory, kept bytecode writes
disabled, left Matplotlib unloaded, and did not broaden strict package reads.
Synthetic primary, synthetic replay, and authoritative bootstrap records all
showed the required modules in `sys.modules` before strict transition.

The authoritative scientific attempt recorded zero strict serialization
`.pyc` reads and zero strict blocked events. Bootstrap/shutdown blocked reads
appearing after the terminal Python exception were traceback source-line
introspection; they did not cause the scientific stop and did not access
protected data.

## R1 status

R1 replaced the generic probe with the full frozen checkpoint schema and exact
writer/reader adapter. Cold and warm probes passed with zero strict imports or
external `.pyc` reads; science remained sealed after a later orchestrator-log
collision.
