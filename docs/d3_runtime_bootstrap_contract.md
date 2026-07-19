# D3 Runtime Bootstrap Contract

Thayer-D3B separates third-party runtime initialization from the future D3
scientific phase. It is a metadata-only and synthetic-runtime contract, not a
D3 experiment.

## Bootstrap phase

Before third-party imports, the scientific launcher redirects temporary files,
package caches, Torch state, and Python bytecode to a fresh runtime subtree.
NumPy, PyTorch, and the required PyTorch submodules then initialize under a
bootstrap guard. Matplotlib is never imported in this interpreter and no
Matplotlib configuration variable is set there. Creation, same-tree rename,
and deletion are permitted only inside the disposable scientific subtree.

Across the separated scientific and postprocessing bootstrap processes, the
audit reproduced the two operations that stopped Thayer-D3R:

- removal of a Matplotlib font-cache lock; and
- removal of a PyTorch temporary-directory probe file.

The exact operations and package routes were recoverable, but D3R did not
persist their Python call stacks. Both paths were confined to its fresh runtime
area and did not name scientific or historical-data artifacts; the missing
frames remain explicitly unresolved.

## Strict scientific phase

After the bootstrap inventory is frozen, the guard enters a strict phase.
Deletion, package-cache writes, recursive traversal, Matplotlib imports,
plotting, historical writes, and nonallowlisted reads are blocked. Exact
scientific modules are compiled in memory from allowlisted source, avoiding
external bytecode-cache reads or writes.

The strict readiness probe performs only a tiny synthetic functional PyTorch
convolution and backward pass. It constructs no neural module or optimizer,
loads no scientific tensor, and executes no decoder forward.

## Shutdown phase

After the readiness status and access log are flushed, shutdown may remove
only package temporary state inside the same disposable runtime subtree.
Descriptor-relative cleanup is resolved to its absolute path before the guard
allows it. Any unresolved or outside-scratch deletion remains terminal.

The readiness contract requires an independent primary process, two cold
processes, one warm-cache process, and one shutdown-audited process. This
contract can authorize only a separately preregistered D3 campaign; it does
not run D3.
