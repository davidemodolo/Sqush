# Contributing

## Layout

```
sqush/         package (see Architecture)
tests/         pytest suite (CPU-only, mocked)
docs/          this documentation (MkDocs Material)
mkdocs.yml     docs config
config.yaml    default runtime config
run.sh         bootstrap + dispatch entry point
```

## Working on the docs

The site is built with MkDocs Material.

```bash
pip install mkdocs-material "pymdown-extensions>=10"
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # what CI runs; fails on broken links/nav
```

Pages live under `docs/` and are wired into `nav:` in `mkdocs.yml`. `--strict` is intentional — a nav entry without a matching file, or a broken internal link, fails the build (and therefore the deploy).

### Deployment

`.github/workflows/docs.yml` builds and deploys to GitHub Pages on every push to `main` that touches `docs/**`, `mkdocs.yml`, or the workflow. It uses the official Pages actions (`upload-pages-artifact` → `deploy-pages`) with `pages: write` + `id-token: write` permissions — no `gh-pages` branch. Set the repo's **Settings → Pages → Source** to **GitHub Actions** once. The site serves at `https://davidemodolo.github.io/Sqush/`.

Trigger a manual run from the Actions tab (`workflow_dispatch`) if needed.

## Working on the code

- Tests must stay CPU‑only and fast — mock anything that would load a model or touch CUDA. Run `python -m pytest` before pushing.
- Keep new code consistent with the surrounding style: dataclasses for config, line‑referenced comments for the tricky quantization/attention math, and log lines that separate prefill vs. decode where relevant.
- When adding a config field, add it to the right `SqushConfig` dataclass, the `config.yaml` example, and (if user‑facing) the [Configuration](../getting-started/configuration.md) docs.

## Conventions worth knowing

- **Session KV correctness** rests on raw‑token splicing + the append‑only fingerprint guard. Don't "simplify" the tail‑render/bridge‑token logic without re‑reading [Session KV reuse](../concepts/session-kv-reuse.md) — the token‑level prefix check is intentionally a tautology in the reuse path.
- **bitsandbytes needs CUDA** to quantize — the LOW‑tier side‑car bake is the only CPU‑friendly quantization path.
