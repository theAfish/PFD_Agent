# NSCF band structure — Python snippet

Uses `MPNonSCFSet.from_prev_calc(mode="line")`. Automatically copies CHGCAR from the SCF directory and sets ICHARG=11.

```python
import json, os, shutil
from datetime import datetime
from pathlib import Path

from pymatgen.io.vasp.sets import MPNonSCFSet

def _work_dir(tag: str) -> Path:
    base = Path(os.environ.get("MATCLAW_SESSION_DIR", ".")) / "vasp"
    d = base / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    d.mkdir(parents=True, exist_ok=True)
    return d

# ── inputs ──────────────────────────────────────────────────────────
SCF_DIRS   = ["<scf_dir1>", "<scf_dir2>"]
SOC        = False
USER_INCAR = {
    "NCORE": 4,
    # "NBANDS": 64,
}

if SOC:
    USER_INCAR.update({
        "LSORBIT": True, "ISPIN": 2,
        "GGA": "PE", "GGA_COMPAT": False, "ISYM": 0,
    })

# ── execution ────────────────────────────────────────────────────────
calc_dirs = []
for scf_dir in SCF_DIRS:
    vis = MPNonSCFSet.from_prev_calc(
        prev_calc_dir=scf_dir, mode="line", user_incar_settings=USER_INCAR,
    )
    d = _work_dir("nscf_kpath")
    vis.write_input(str(d))   # copies CHGCAR automatically
    if SOC and (Path(scf_dir) / "WAVECAR").exists():
        shutil.copy2(Path(scf_dir) / "WAVECAR", d / "WAVECAR")
    calc_dirs.append(str(d.resolve()))

print(json.dumps({"status": "success", "calc_dir_list": calc_dirs}))
```

**MPNonSCFSet line-mode defaults:** ICHARG=11, LCHARG=False, LWAVE=False, LORBIT=11. k-path from `HighSymmKpath` (64 points per segment).
