# Read results — Python snippet

Parse a completed calculation directory. Set `CALC_TYPE` to match the job that was run.

```python
import json
from pathlib import Path
from pymatgen.io.vasp.outputs import Vasprun

# ── inputs ──────────────────────────────────────────────────────────
CALC_TYPE = "<relaxation|scf|nscf>"
CALC_DIR  = "<calc_dir>"

# ── execution ────────────────────────────────────────────────────────
d  = Path(CALC_DIR)
vr = Vasprun(str(d / "vasprun.xml"), parse_dos=False, parse_eigen=(CALC_TYPE == "nscf"))
result = {"status": "success", "calc_type": CALC_TYPE}

if CALC_TYPE == "relaxation":
    forces = vr.ionic_steps[-1]["forces"]
    result.update({
        "structure":    vr.final_structure.as_dict(),
        "total_energy": vr.final_energy,
        "max_force":    max(sum(f**2 for f in force)**0.5 for force in forces),
        "stress":       vr.ionic_steps[-1].get("stress"),
        "ionic_steps":  len(vr.ionic_steps),
    })
elif CALC_TYPE == "scf":
    gap, cbm, vbm, is_metal = vr.eigenvalue_band_properties
    result.update({
        "structure":    vr.final_structure.as_dict(),
        "total_energy": vr.final_energy,
        "efermi":       vr.efermi,
        "band_gap":     gap,
        "is_metal":     is_metal,
    })
elif CALC_TYPE == "nscf":
    bs = vr.get_band_structure(line_mode=True)
    gap, cbm, vbm, is_metal = vr.eigenvalue_band_properties
    result.update({
        "structure": vr.final_structure.as_dict(),
        "efermi":    vr.efermi,
        "band_gap":  gap,
        "cbm":       cbm,
        "vbm":       vbm,
        "is_metal":  is_metal,
        "band_structure_summary": {
            "n_kpoints": len(bs.kpoints),
            "n_bands":   bs.nb_bands,
            "labels":    {str(k): list(v) for k, v in bs.labels_dict.items()},
        },
    })

print(json.dumps(result))
```
