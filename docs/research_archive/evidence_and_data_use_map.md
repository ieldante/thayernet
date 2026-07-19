# Evidence and data-use map

## What will be done with all of this data?

The program uses data in four distinct roles: inputs to construct controlled
observations; hidden truth used only to score or diagnose; compact metric and
endpoint evidence used to support claims; and operational provenance used to
replay or audit those claims. Raw and dense data remain local. Git contains the
code, contracts, hashes, compact tables, authoritative reports, and selected
figures needed to understand and re-establish the evidence chain.

| Data category | Source and format | Scale / raw or derived | Scientific role | Commit? | Needed to reproduce? | Poster / paper / supplement / Phase II use | Retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Raw Galaxy10 DECaLS | `data/Galaxy10_DECals.h5`; HDF5, 2.735 GB; raw external dataset | One local source file plus an exact duplicate in `data_exploration/data/` | Early synthetic cutout benchmark and scene construction | No | Yes for early benchmark replay; loader and hash suffice in Git | Background only / dataset methods / data caveat / not the primary Phase II structured set | Local read-only source; retain one verified canonical copy and document the duplicate; do not delete during this audit |
| CatSim/BTK source catalogs | Local catalogs and HDF5/Parquet; raw external/simulator inputs | Multi-GB catalogs and scene containers | Controlled g/r/z simulated galaxy rendering and independent-scene generation | No | Yes, via source/version/hash/loader contract | Simulation schematic / controlled-inference contract / generator details / basis for fresh validation sampling | Local; preserve source license/provenance and hashes |
| DR10 cutouts and triplets | `data/dr10_grz_cutouts/`, `data/dr10_scene_triplets/`; FITS/images; raw downloaded | Local survey probes | OOD/provenance foundation; no closed structured-solver performance claim | No | Only for real-data follow-up | Limitation / future real-data closure / supplement provenance / possible later robustness set | Local; manifests and download scripts in Git |
| Protected development tensors | HDF5 scene partitions under ignored run trees; derived | Training/validation/development partitions | Model training and frozen development evaluation | No | Needed only for exact historical replay | Not shown as raw / partition methods / leakage safeguards / excluded from Phase II independent set | Local protected; access by frozen manifests only |
| Atlas tensors | Atlas HDF5, embeddings, candidate arrays; derived/protected diagnostic | Tens to hundreds of MB | Ambiguity witnesses and diagnostic family comparison | No | Compact report/table sufficient for claims; dense tensors needed for exact replay | One ambiguity example / diagnostic methods / detailed failure analysis / never reused as independent validation | Local protected; retain hashes and access rules |
| Lockbox / future final set | Sealed or absent; protected | No authorized outcome data | Reserved final evaluation | No | Not accessed in Phase I | Explicitly state sealed / protocol / access audit / excluded from all tuning | Remain sealed; no tensor or outcome enters Git |
| Hidden isolated source truth | Simulator-isolated requested/companion images and catalog parameters; HDF5/arrays | Dense truth, available because scenes are simulated | Supervision and evaluation only unless an oracle experiment explicitly says otherwise | No | Needed for metric replay, not normal inference | Truth-versus-inference diagram / evaluation contract / oracle caveat / excluded from Phase II inference | Local protected; never stage tensors |
| Generated blended observations | BTK blends, same-PSF pairs, PSF-diverse pairs; HDF5/NPY/NPZ | Many GB; derived | Scientific fitting inputs and deterministic replay | No | Yes for exact replay; hashes/manifests define identity | Representative image only / acquisition contract / replay details / generate fresh independent scenes | Local append-only; selected compact PNGs only in archive |
| Synthetic unit fixtures | Small arrays created inside tests | Tiny; derived | Numerical, schema, guard, and physical-contract tests | Yes when embedded in source/tests and not protected | Yes | Usually none / implementation assurance / supplement tests / reusable Phase II validation | Commit with tests |
| Neural reconstructions | HDF5/NPY outputs and checkpoints | Hundreds of MB to GB; derived | Accuracy, promptability, candidate diversity, and safety labels | Raw outputs: no. Compact metrics: yes | Dense outputs required for exact historical replay | A few examples / aggregate metrics / model-family supplement / not independent validation evidence | Local; compact tables and selected figures archived |
| Structured-solver source estimates | Fit records and source layers; JSON plus dense arrays | Small JSON to large endpoint arrays; derived | Endpoint multiplicity, flux allocation, morphology, and reconstruction diagnostics | Compact JSON/CSV summaries only | Endpoints and replay hashes are sufficient for most audits; dense images remain local | One ambiguity panel / primary identifiability tables / endpoint supplement / new fits on independent scenes | Local raw, compact summaries committed |
| Multi-start endpoints | CSV/JSON endpoint histories | Hundreds to thousands of endpoints; derived | Global identifiability, basin multiplicity, convergence, clustering | Compact endpoint/class tables: yes; full histories: no | Yes for global result replay | Multiplicity diagram / methods and primary outcomes / full compact table / equalized starts in Phase II | Preserve local full histories; archive representative tables |
| Singular values, Jacobians, Hessians, ranks, conditions | CSV/JSON/plots plus possible dense matrices | Compact spectra to dense matrices; derived | Local identifiability and local/global comparison | Summaries and selected plots yes; dense matrices no | Summary spectra/rank plus code are adequate for claim audit | Conditioning-without-uniqueness plot / local-global distinction / numerical diagnostics / frozen endpoints | Local dense matrices; committed metrics/plots |
| Oracle per-source photometry | Exact g/r/z source flux from isolated truth | Tiny tables; evaluation/oracle intervention | Tests conditional uniqueness under information unavailable to ordinary inference | Compact table/report yes | Yes to reproduce oracle contract | 7/8 conditional result / oracle ablation caveat / full table / discovery-only comparator | Commit compact measurement contract, never isolated tensors |
| External total photometry | Exact noisy measurement table with frozen 5% uncertainties | Tiny CSV; simulated measurement intervention | Test whether one additional total-flux constraint reduces multiplicity | Yes | Yes | Helpful/not-helpful table / controlled intervention / uncertainty contract / primary Phase II intervention | Commit table and protocol |
| External per-band photometry | Frozen per-band measurement table with 5% uncertainties | Tiny CSV; simulated intervention | Secondary comparison with total photometry | Yes | Yes | Usually paper/supplement / comparison / full table / secondary Phase II endpoint only | Commit compact table |
| Scene-stratification features | Compact CSV derived post-fit from truth/catalog and existing metrics | Eight rows; exploratory | Explain heterogeneity and generate a validation hypothesis | Yes, with truth-derived warning | Yes for discovery analysis | Candidate rule / exploratory analysis / full feature table / freeze strata, do not treat as deployable inputs | Commit; never use discovery outcomes to tune Phase II threshold |
| Audit labels | Per-query safe/unsafe and query-state labels | Thousands of rows; truth-derived | Train/evaluate PRE/POST research classifiers | Raw label tables no when bulky; compact prevalence/metrics yes | Historical replay needs local tables | Audit flow / methods / label-collapse supplement / Phase II policy only after nondegenerate reconstruction outputs | Local raw; commit prevalence and gate tables |
| Checkpoints and optimizer state | `outputs/checkpoints/` and ignored run trees; PT/PTH | 769 checkpoint-like files in initial broad scan; 134 MB in top-level checkpoint directory and more in runs | Historical model replay and integrity verification | No | Needed only for exact historical replay | None / model versions and hashes / checkpoint integrity / not a Phase II artifact unless a frozen model is reused | Local; retain hash inventories and role; do not stage |
| Reports and protocols | Markdown/JSON | Compact; derived authority | Human-readable scientific authority and frozen decision rules | Yes | Yes | Direct source for all products | Commit all selected authorities through curated archive |
| Compact tables and manifests | CSV/JSON | KB to low MB; derived authority/provenance | Exact values, scene identities, hashes, gates, and decisions | Yes after privacy/protection review | Yes | Tables/claims / results / supplement / freeze inputs and thresholds | Commit selected compact copies |
| Figures | PNG | 851 initial research figures, mostly intermediate | Visual evidence and communication | Only selected compact central figures | Not generally; regeneration code and tables preferred | Classified below | Keep all local; commit only curated set |
| Logs and caches | LOG/TXT/JSONL, pycache, replay caches | From KB to multi-GB; derived operational material | Debugging, access audit, acceleration | No, except compact validation summaries | Not for scientific interpretation | None / none / selected integrity summary / regenerate | Local; caches are disposable only under a separate retention decision, not this audit |

## Figure disposition

| Figure | Disposition | Use |
| --- | --- | --- |
| Coordinate prompt-swap flagship grid | Local poster candidate; excluded from Git pending protected-truth publication review | Establishes promptability visually but embeds isolated truth panels. |
| Unrestricted or flux-free recoverability frontier | Main paper / poster | Shows the source-allocation problem and strict nonoracle outcome. |
| PSF S1–S2–P2 frontier and condition heatmap | Main paper | Separates information geometry from uniqueness and includes the same-PSF control. |
| External-photometry corrected information-source comparison | Poster / main paper | Central 4-helpful/4-not-helpful scene result. |
| Corrected decision tree | Supplement only | Exploratory low-|ΔB/T| rule; must be labeled discovery-set and truth-derived. |
| Family-E1P prompt-conditioning diagnosis | Supplement | Shows prompt modulation surviving without identity alignment. |
| Loss-gradient heatmap | Supplement / methods | Supports objective-misalignment mechanism. |
| Audit risk-coverage curve | Supplement | Demonstrates zero usable safe coverage; not a deployment figure. |
| Training curves, repeated contact sheets, endpoint-per-start panels | Archive-only | Useful for forensic replay but redundant for primary narrative. |
| Dense galleries, raw observations, duplicated figures | Exclude from Git | Bulky, protected, or redundant. |

## Poster use

- Information-contract ladder from unrestricted output to structural and
  measurement interventions.
- Oracle-information `7/8` versus nonoracle flux-free `0/8`, explicitly marked
  as a cross-contract transition rather than a clean one-factor ablation.
- PSF composite geometry improvement in `15/16` fits alongside `0/8` strict
  uniqueness and the S2 causal breakdown.
- External total-photometry response: helpful Scenes `0, 5, 51, 73`; not
  helpful `3, 6, 18, 81`.
- PRE → structured fit → global identifiability → RECONSTRUCT / ACQUIRE /
  DON'T EVEN TRY → future POST flow.
- One publication-cleared allocation witness and one multi-start endpoint
  example; do not export hidden-truth panels without explicit review.

## Paper use

- Controlled inference contract, protected-set boundary, and truth-use rules.
- Promptability-versus-recoverability distinction.
- Unrestricted null-space proof and the local/global identifiability split.
- Structural/oracle, flux-free, same-PSF S2, PSF-diverse P2, and external-
  photometry contracts as a qualified intervention ladder.
- External-photometry heterogeneity as discovery evidence, including unequal
  start-count limitation and truth-derived stratification features.
- S1/S2/P2 acquisition control, convergence corrections, replay, and integrity.
- Limitations, prohibited claims, and independent validation plan.

## Supplement use

- Direct/residual/balanced model history and grouped-data correction.
- Recoverability heads, hierarchical calibration, PU/PF/MH/ME failures.
- Full loss-geometry, output parameterization, D0–D3, and D3 engineering trail.
- Family-E physical-contract failure and signed-residual correction.
- PRE/POST direct-audit failure, all-unsafe label collapse, and checkpoint
  integrity.
- Full compact scene, endpoint, rank, condition, and feature tables.

## Phase II use

The eight Phase I scenes may be used only to freeze the hypothesis, primary
metric, thresholds, starts, solver budget, runtime assumptions, and sampling
strata. They may not appear in the independent validation sample. Total source
photometry is the primary intervention; flux-free S1 is the baseline and P2 is
the comparator. The discovery low-|ΔB/T| threshold may be evaluated exactly as
frozen but not retuned, and because it is truth-derived it is an explanatory
validation endpoint rather than an operational policy feature.

## Retention and deletion boundary

This audit deletes nothing. Local raw data, protected tensors, checkpoints,
generated observations, dense outputs, and full logs remain in place. A future
storage-reduction action would require a separate, explicit retention decision
after verifying hashes and replay coverage; it is not authorized by this map.
