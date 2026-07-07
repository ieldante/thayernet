# GitHub Release Checklist

Before pushing this repository to a public remote, verify:

- No dataset files are tracked.
- Notebook outputs and execution counts are cleared.
- No personal paths, usernames, hardware identifiers, or local machine paths remain.
- No caches are present, including `__pycache__/`, `.ipynb_checkpoints/`, and `.DS_Store`.
- `.venv/` is ignored and untracked.
- Large generated outputs, checkpoints, figures, and draft PDFs are not staged.
- `git status` is clean except for intended files before committing.
- `git ls-files` has been reviewed.
- `data/.gitkeep` and `reports/.gitkeep` are tracked, but dataset and generated report artifacts are not.
- Privacy grep has been run against the repository excluding `.git`, `.venv`, `data`, and generated caches.
