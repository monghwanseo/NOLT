import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from src.data import config as cfg

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def fmt_size(n: int) -> str:
    for u in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"

def main():
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    sources = []
    for p in [cfg.RAW_XLSX, cfg.RAW_XLSX_BACKUP]:
        if p.exists():
            sources.append({
                'name': p.name,
                'size': p.stat().st_size,
                'sha256': sha256(p),
            })

    outputs = []
    for p in sorted(cfg.PROCESSED_DIR.glob('*')):
        outputs.append({
            'name': p.name,
            'size': p.stat().st_size,
            'sha256': sha256(p),
        })

    md = f"""# NOLT Data Freeze -v1

**Frozen:** {ts}

## DO NOT MODIFY

This snapshot is the reference for all downstream NOLT experiments. Any change to
`Bloomberg/*.xlsx` or to the produced files in `data/processed/` requires a new
freeze (v2, v3, ...) and re-running all dependent experiments.

If raw data needs to be updated, do NOT overwrite `Bloomberg/spx_2.xlsx` -preserve the original and add a new versioned file (e.g. `spx_3.xlsx`).

## Source files

| File | Size | SHA-256 |
|---|---|---|
"""
    for s in sources:
        md += f"| `Bloomberg/{s['name']}` | {fmt_size(s['size'])} | `{s['sha256']}` |\n"

    md += f"""
- **Primary**: `spx_2.xlsx` (57 sheets: 'spx' + 1..56)
- **Backup**: `spx_1.xlsx` (28 sheets -superset by spx_2; not used in pipeline)

## Processed outputs (data/processed/)

| File | Size | SHA-256 |
|---|---|---|
"""
    for o in outputs:
        md += f"| `{o['name']}` | {fmt_size(o['size'])} | `{o['sha256']}` |\n"

    md += f"""
## Standards (LOCKED)

- Risk-free rate: **r = {cfg.R}**
- Dividend yield (BSM/PCP/Greeks): **q = q_implied (per-date) | {cfg.Q_BASELINE} (M10 fallback)**
- Random seed (all stochastic ops): **{cfg.SEED}**
- Train/Val/Test split: **{cfg.TRAIN_VAL_TEST[0]:.0%} / {cfg.TRAIN_VAL_TEST[1]:.0%} / {cfg.TRAIN_VAL_TEST[2]:.0%}**
- q calibration: cross-maturity ratio with |tau1 - tau2| >= {cfg.Q_MIN_DTAU} years
- **q = 0 is forbidden** (only sanctioned in Step 6 PCP bootstrap; see `memory/feedback_q_baseline.md`)

## Quality classification thresholds

```
{cfg.QUALITY_THRESHOLDS}
```

- **FULL_USE**: IV/Delta/Gamma >= 0.95 AND Mid Price >= 0.80
- **GREEKS_ONLY**: IV/Delta/Gamma >= 0.95 AND Mid Price < 0.80
- **EXCLUDE**: any of IV/Delta/Gamma < 0.95

## Window definitions

| Window | Selection | Use |
|---|---|---|
| **A** | expiries {{06/18/26, 12/18/26}} -(FULL_USE -GREEKS_ONLY) | Primary (M5-M10 reproduction) |
| **B** | all (FULL_USE -GREEKS_ONLY) | Extended cross-section |
| **C** | (FULL_USE -GREEKS_ONLY) AND date_span >= {cfg.WINDOW_C_MIN_DAYS} days | Long-history (latent state) |

## Regeneration

```bash
# Step 1: prepare panels + run sanity (Steps 2-7)
python scripts/01_build_data.py

# Step 2: regenerate quality report + figures (Step 8)
python scripts/02_quality_report.py

# Step 3: re-freeze (Step 9, this script)
python scripts/03_freeze.py
```

If `data/processed/` is deleted, re-running step 1 reproduces it deterministically
(seed = {cfg.SEED}). Compare regenerated SHA-256 against this freeze to confirm
byte-identical reproduction.
"""

    out = cfg.DATA_DIR / 'data_freeze_v1.md'
    out.write_text(md, encoding='utf-8')
    print(f"-> {out}")
    print(f"   {len(sources)} source files + {len(outputs)} processed outputs hashed")

if __name__ == '__main__':
    main()
