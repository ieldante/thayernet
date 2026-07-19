# Pre-commit validation

## Scope and decision rule

This audit validates the canonical Thayer archive without rerunning a
scientific campaign. It covers repository state, authority reconciliation,
archive provenance, compact schemas, links, source syntax, tests, secrets,
protected data, checkpoints, large files, duplicate selection, and the final
Git index. Environmental and historical-integrity failures are preserved below
rather than hidden.

**Scientific campaigns rerun:** none.

**Current decision:** `PASS_FOR_SINGLE_CANONICAL_COMMIT`.

## Repository protection

- Pre-audit branch: `thayer-select`.
- Pre-audit HEAD: `74b8ff7efbbf7e9891cc8fd8095a9931e3b63174`.
- Backup branch: `backup/pre-thayer-canonical-archive-20260719T215720Z`, created
  at that HEAD without switching or pushing.
- Initial Git index: empty.
- Initial tracked modifications: 5; initial nonignored untracked files: 394.
- Local branches `thayer-br-0.2` and `thayer-br-0.3` were merged with gone
  upstreams; inspected but not deleted.
- No local output, checkpoint, dataset, cache, or historical run directory was
  deleted or rewritten.

The compact inventory deliberately hashes every file above 5,000,000 bytes
and every selected archive source/copy, not all 35,060 small local research
files. Small-file identity is supplied where scientifically relevant by run
manifests, campaign checkpoint inventories, and archive provenance. This scope
boundary avoids implying a nonexistent 35,060-row hash manifest.

## Authority and claim validation

- Central outcomes were reconciled using final report → manifest → frozen
  protocol → correction → compact table → source/test ordering.
- The supersession ledger contains 21 field-scoped resolutions.
- The experiment ledger contains 124 run roots plus the repository-level
  null-space campaign. Forty-seven claim-bearing run roots, including every
  provenance-backed curated run, have explicit scientific metadata; 42
  repeated launch/D3 records have explicit grouped engineering-history
  metadata; 35 minor or incomplete records remain honest locators rather than
  fabricated reconstructions.
- All 34 provenance-backed run roots are covered by substantive overrides.
- The first flux-free launch is typed engineering-invalid and supplies no
  scientific 0/8 result; the valid later run governs.
- No report title is silently used as a final outcome when an outcome section
  cannot be safely extracted.
- Cross-contract wording preserves the non-clean oracle-versus-flux-free
  ablation and the S2 qualification of the P2 15/16 composite result.

## Curated archive validation

Final archive snapshot:

| Check | Result |
| --- | --- |
| Campaign directories | 34 |
| Files / bytes | 174 / 2,331,401 |
| Extensions | 42 CSV, 15 JSON, 100 Markdown, 14 PNG, 3 Python |
| Maximum file | 201,081 bytes |
| Builder allowlist/tree agreement | PASS; 0 missing or unexpected |
| Strict JSON | PASS; 15/15; nonfinite tokens rejected |
| CSV syntax and uniform widths | PASS; 42/42 |
| PNG signature/trailer | PASS; 14/14 |
| Provenance documents / rows | 34 / 139 |
| Original/archive size and SHA-256 | PASS; 139/139 |
| Declared byte-exact/path-token transformations | PASS; 139/139 |
| Non-provenance coverage | PASS; exactly one provenance row per selected artifact |
| Banned types or file above 5 MB | 0 |
| Duplicate hashes inside archive | 0 |
| Machine path, email, or high-confidence secret | 0 |

Seven initially considered images were excluded fail-closed: five isolated or
hidden truth-panel grids, one development blend/output gallery, and one
noncentral training-microset output grid. Originals remain local and unchanged.

## Compact schema and link validation

- Candidate canonical data contained 15 strict JSON and 44 CSV files outside
  excluded `data_exploration/`; all parsed and every CSV had a uniform row
  width.
- The pre-index Markdown audit covered 265 modified/untracked candidates
  outside `data_exploration/`: 129 local links, 0 broken, 0 absolute-path
  links, and 0 links into ignored `outputs/runs`. The two audit documents
  created after that scan introduce no local Markdown links; the final index
  contains 267 Markdown files.
- The canonical-map appendix contains exactly 124 run-directory rows; the
  machine ledger also contains the non-run analytic campaign.

## Source, privacy, and command audit

- AST parsing covered 348 Python files across repository source, scripts,
  tests, and archive drivers: 0 syntax failures. A final staged-candidate AST
  count is recorded below after index construction.
- No high-confidence credential, private key, bearer token, email address, or
  private URL was found in the intended commit.
- Candidate `/Users/` literals outside the excluded notebook are privacy/path
  scanner tests, not active machine-specific paths. Curated copies tokenize
  source machine paths and record both hashes.
- No `shell=True`, `os.system`, broad `rm -rf`, hard reset/clean, or force-push
  command was found. Scoped deletion code targets runtime scratch/atomic temp
  files or pytest `tmp_path` only.
- Dynamic `exec(compile(...))` in the D3 frozen-source loaders was reviewed as
  intentional elevated-complexity contract code; it does not execute in this
  archive audit.
- The executed `data_exploration/` notebook, embedded outputs, personal paths,
  and figures are excluded in full.

## Test results

No test failure below changed a scientific result or justified altering frozen
campaign code.

1. A broad default pytest collection found 703 tests and 18 collection errors.
   Pytest traversed ignored historical run copies with duplicate module names;
   repository collection also lacked optional `requests` in the BTK Python
   environment and one required artifact-directory environment variable. This
   is an honest full-workspace collection failure.
2. Focused repository tests in the historical BTK environment, excluding the
   two environment-selected files, produced **519 passed, 18 failed, 6
   skipped**. The 18 failures were fully classified:
   - 2 D3 entrypoint tests correctly detected the intentionally changed README
     against a historical frozen source manifest;
   - 9 FITS-blend tests used `zip(strict=True)`, unsupported by the historical
     Python 3.9 interpreter;
   - 1 signed-residual assertion was numerically sensitive under historical
     PyTorch/Python;
   - 6 Family-E1 artifact tests lacked `THAYER_FAMILY_E1_RUN`.
3. Under the current Python 3.14 environment, the FITS-blend,
   signed-residual, and offline DR10 downloader selections all passed:
   **32 passed**.
4. With `THAYER_FAMILY_E1_RUN` set to its frozen local run, the six Family-E1
   artifact tests passed: **6 passed**.
5. With `THAYER_AUDIT_RUN_DIR` set, the direct-audit artifact suite produced
   **7 passed, 1 failed**. The sole failure is the historical README SHA-256,
   expected because this task is explicitly required to update README. The
   historical provenance record was not rewritten.

The root environment is not locked: `requirements.txt` has 15 unpinned
dependencies, `pyproject.toml` contains formatter/linter configuration rather
than a project environment, and no lockfile exists. Test interpretation must
therefore name the environment as above.

## Large-file, checkpoint, and protection audit

- `large_file_audit.csv` records all 340 files above 5,000,000 bytes; 88 exceed
  50,000,000 and 75 exceed 100,000,000. All 340 are Git-ignored.
- The initial broad scan found 769 checkpoint-like files. Checkpoints,
  optimizers, semantic states, dense tensors, generated observations, caches,
  and logs remain local; no such type is intended for the index.
- The two 2,735,267,419-byte Galaxy10 HDF5 copies and the
  6,291,456,128-byte replay cache remain untouched and unstaged.
- No Git LFS configuration was introduced.
- Two byte-identical source/package pairs are intentional, documented archive
  exceptions: the live hashed null-space report and its curated package copy,
  and the already-tracked grouped-correctness report and its curated package
  copy. No other exact duplicate is selected.

## Final staged-set validation

The final proposed index contains 585 files: 579 additions and 6
modifications, with no staged deletion or rename. It contains 110,971 textual
additions, 448 textual deletions, and 14 binary PNG files. Its aggregate blob
size is 7,983,614 bytes and its largest file is 201,081 bytes. Therefore no
staged file exceeds 5 MB, 50 MB, or 100 MB.

The self-excluding commit manifest inventories 584 other files totaling
7,865,618 bytes. It records each path, category, byte size, SHA-256, and reason
for inclusion. Its category counts sum to 584, and its canonical staged-record
digest covers the path, size, and content hash of every listed file. The
manifest explains why its own exact blob hash cannot be embedded in itself.

Final index checks:

- File types are 267 Markdown, 245 Python, 44 CSV, 15 JSON, and 14 PNG; no
  banned or unexplained suffix is staged.
- All 245 staged Python files parse as ASTs. All 15 JSON files parse strictly;
  all 44 CSV files parse with uniform row widths; all 14 PNGs have valid
  signatures and trailers.
- No path under `data/`, `data_exploration/`, or `outputs/` is staged. Raw
  datasets, notebooks, checkpoints, tensor arrays, caches, and run logs remain
  outside the index.
- The final secret/private-key/email/private-URL scan has zero
  high-confidence hits. Literal `/Users/` strings occur only in scanner test
  cases and the audit text that documents them, never as active paths.
- The only exact duplicate among staged blobs is the deliberate
  source/package pair `reports/thayer_recoverability_v0.md` and
  `docs/experiment_archive/recoverability_nullspace/final_report.md`. The live
  source is hashed by code and the curated archive must be self-contained;
  `artifact_index.md` documents this exception.
- Cached name-status and full diff review found no deletion, no conflict
  marker, and no unexplained scope expansion. After this commit, the only
  remaining nonignored working-tree material is the nine-file protected
  `data_exploration/` directory.
- `git diff --cached --check` reports 2,142 trailing-whitespace lines in 52
  preserved historical/evidence files and 29 new-blank-EOF warnings. Most are
  CRLF-marked CSV evidence; the remainder are byte-preserved source reports,
  protocols, and campaign code. Normalizing them would break recorded
  provenance or silently alter historical artifacts. These are classified as
  formatting-only preservation warnings, not merge markers or executable
  defects.

The staged set therefore passes the repository, scientific-authority,
protection, size, schema, syntax, provenance, and commit-scope gates for the
single canonical archive commit. The environmental and historical test
limitations above remain explicit rather than being converted into a false
all-tests-pass claim.

## Remaining authority and reproducibility gaps

- No independent-scene, population-prevalence, or real-survey validation
  authority exists.
- No operational observable replaces truth-derived/post-fit `|ΔB/T|`.
- No direct reconstruction-accuracy authority exists for an independently
  unique nonoracle target.
- POST has no nondegenerate safe-positive population.
- Full neural/structured replay needs excluded local arrays and checkpoints.
- Nineteen tests reference `outputs/runs`; 21 need environment-selected/local
  artifacts. Seven D3 modules have no direct textual test reference, though
  they may be tested transitively.
- Full D3 eight-scene/capacity branches remain unknown.
- The experiment ledger intentionally leaves 35 minor/incomplete run records
  as metadata locators where compact authority does not support full
  extraction.
