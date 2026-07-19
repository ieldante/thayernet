# Allowlisted File-Access Contract

The repository-integrity campaign used a fail-closed exact-path contract.

- The initial allowlist contained only authoritative reports,
  preregistrations, inventories, the fresh run, and the guarded launcher.
- Each expansion was derived from already approved authoritative text, a closed
  AST import edge, or the sole authoritative checkpoint inventory.
- Wildcards, globbing, recursive discovery, directory walking, historical
  writes, rename, removal, and nonapproved external writes were denied.
- Every Python open, stat, directory-iteration, and write decision was logged
  by the audit hook.
- Shell work used explicit paths and was recorded in the campaign command log.
- A blocked decision never became a successful access.

Benign library attempts to inspect fonts, system metadata, caches, or temporary
cleanup paths were denied and preserved in the blocked-access log. The guard
was not relaxed to accommodate them. Scientific execution remained restricted
to the exact frozen training scene and target row; Atlas, development, and
lockbox data were never opened.

## Runtime lifecycle extension

Thayer-D3B adds explicit bootstrap, strict, and shutdown phases. Bootstrap may
create, rename, or delete only inside a fresh disposable runtime subtree.
Strict phase blocks all deletion and cache writes and permits only exact reads
plus preregistered new-run outputs. Shutdown cleanup begins only after readiness
evidence is flushed and may remove only resolved paths under runtime scratch.

The readiness launcher compiles exact scientific source in memory, preventing
legacy bytecode-cache reads during strict execution. Its final cold, warm, and
post-shutdown processes had zero strict blocked accesses and zero protected
access. Matplotlib is absent from every scientific interpreter and confined to
the separate postprocessor. The authoritative closure also compares frozen
bootstrap and strict-end inventories and audits all postprocessor lifecycle
paths against its own disposable root.
