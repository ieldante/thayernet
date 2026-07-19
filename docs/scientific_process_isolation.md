# Scientific Process Isolation

The D3 readiness scientific launcher runs in a dedicated Python interpreter
with `-B` and `PYTHONDONTWRITEBYTECODE=1`. Its temporary, cache, configuration,
Torch, and pycache roots are fresh and process-specific. `MPLCONFIGDIR` and
`MPLBACKEND` are absent from this interpreter.

The bootstrap phase imports NumPy, PyTorch, and exact required PyTorch
submodules before the strict guard activates. The strict phase then compiles
the exact plotting-free project modules in memory, inspects model and
scientific function definitions without instantiating them, and performs only
synthetic functional tensor operations. It cannot delete files, write package
caches or bytecode, import plotting code, deserialize a scientific tensor,
instantiate a model, construct an optimizer, or execute a decoder forward.

Postprocessing is never performed in the scientific interpreter. A separate
launcher, runtime root, access log, and process handle synthetic Matplotlib/Agg
readiness. The scientific readiness marker is valid only from the plotting-free
scientific process. The authoritative run recorded initial, bootstrap,
strict-end, and post-shutdown inventories for the primary, cold, warm-cache,
and shutdown-audited processes and proved that strict execution changed none of
their runtime files.
