# D3 Scientific Dependency Schema

The D3 scientific contract contains 97 required dependencies. The earlier
21-row D1R table remains valid, but it is an artifact/runtime prerequisite set,
not a complete scientific dependency schema. It did not directly package the
sky vector or plausibility values.

The Thayer-D3C inventory covers observation configuration, forward
plausibility, truth coverage, the square-output contract, prompt and hard
assignment semantics, exact code/runtime hashes, four immutable scientific
artifact references, and row identity. Every row records type, shape, units,
band order, consumer, source symbol, exact source path/key/hash, default policy,
cross-source status, and resolution status.

The complete machine-readable inventory is
`tables/d3_scientific_dependency_inventory.csv` inside the authoritative
Thayer-D3C run. All 97 rows resolved, no default is permitted, and the AST and
schema audit found zero hidden dependencies.

