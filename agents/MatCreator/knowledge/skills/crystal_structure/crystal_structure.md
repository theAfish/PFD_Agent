---
name: crystal_structure
description: Build, inspect, modify, and curate diverse atomic structures using ase-based tool.
tags: [crystal structure]
tools: [run_bash,run_python_file]
dependent_skills: []
---

## Overview

Use the CLI script `crystal_structure_tools.py` (located alongside this file) to build, inspect, transform, and curate atomic structure files. Every command writes a JSON object to stdout and exits **0** on success or **1** on error. Always parse the JSON output to retrieve the output file path and confirm `"status": "success"`.

```
python skills/crystal_structure/crystal_structure_tools.py <command> [options]
```

**Supported output formats** (pass to `--output-format`): `extxyz` (default), `xyz`, `cif`, `vasp`, `json`.

**Auto-generated paths**: when `--output` is omitted the script creates a timestamped working directory and places the output file inside it. The absolute path is always returned in the JSON field `structure_path`.

---

## Commands

### 1. `build-bulk-crystal`

Create a primitive (or supercell) bulk crystal from a chemical formula and a crystal-structure prototype using ASE's `bulk()` builder.

**Positional arguments**
| Argument | Description |
|---|---|
| `formula` | Chemical formula, e.g. `Al`, `NaCl`, `MgO` |
| `crystal_structure` | Prototype name: `fcc`, `bcc`, `hcp`, `rocksalt`, `zincblende`, `diamond`, `cesiumchloride`, `fluorite`, `wurtzite`, etc. |

**Optional arguments**
| Flag | Type | Description |
|---|---|---|
| `--a` | float | Lattice constant *a* (Å) |
| `--c` | float | Lattice constant *c* (Å) |
| `--covera` | float | *c/a* ratio (used for hcp-like structures) |
| `--u` | float | Internal coordinate *u* |
| `--spacegroup` | int | Spacegroup number (for generic structures) |
| `--basis` | JSON | Fractional basis positions as a JSON list of `[x,y,z]` lists |
| `--orthorhombic` | flag | Request an orthorhombic cell |
| `--cubic` | flag | Request a cubic cell |
| `--size` | int | Uniform supercell repetition (default: 1); ignored when `--size-matrix` is given |
| `--size-matrix` | JSON | `[nx,ny,nz]` or `[[m00,m01,m02],…]` 3×3 integer matrix |
| `--vacuum` | float | Vacuum padding in Å added on all sides |
| `--output-format` | str | Output format (default: `extxyz`) |
| `--output` | path | Explicit output file path |

**JSON response fields**
```
status, message, structure_path, chemical_formula, num_atoms, cell, pbc
```

**Examples**
```bash
# FCC aluminium, default lattice constant
python skills/crystal_structure/crystal_structure_tools.py build-bulk-crystal Al fcc

# BCC iron with explicit lattice constant, 2×2×2 supercell
python skills/crystal_structure/crystal_structure_tools.py build-bulk-crystal Fe bcc \
    --a 2.87 --size 2

# HCP titanium with c/a ratio, output as CIF
python skills/crystal_structure/crystal_structure_tools.py build-bulk-crystal Ti hcp \
    --a 2.95 --covera 1.586 --output-format cif --output Ti_hcp.cif

# NaCl rocksalt using a 3×3 supercell matrix
python skills/crystal_structure/crystal_structure_tools.py build-bulk-crystal NaCl rocksalt \
    --size-matrix "[[3,0,0],[0,3,0],[0,0,3]]"
```

---

### 2. `build-supercell`

Expand an existing structure file into a supercell.

**Positional arguments**
| Argument | Description |
|---|---|
| `input` | Path to any ASE-readable structure file |

**Optional arguments**
| Flag | Type | Description |
|---|---|---|
| `--size` | int | Uniform repeat (default: 1); ignored when `--size-matrix` is given |
| `--size-matrix` | JSON | `[nx,ny,nz]` or 3×3 integer matrix |
| `--output-format` | str | Output format (default: `extxyz`) |
| `--output` | path | Explicit output file path |

**JSON response fields**
```
status, message, structure_path, chemical_formula, num_atoms, cell, pbc
```

**Examples**
```bash
# 2×2×2 supercell from an existing extxyz file
python skills/crystal_structure/crystal_structure_tools.py build-supercell Al_fcc.extxyz \
    --size 2

# Anisotropic 3×3×1 supercell
python skills/crystal_structure/crystal_structure_tools.py build-supercell structure.cif \
    --size-matrix "[3,3,1]" --output-format vasp --output POSCAR
```

---

### 3. `perturb-atoms`

Generate multiple perturbed replicas of a structure by randomly distorting the cell and displacing atoms. Outputs a **multi-frame** file.

**Positional arguments**
| Argument | Description |
|---|---|
| `input` | Path to input structure file |

**Required flags**
| Flag | Type | Description |
|---|---|---|
| `--pert-num` | int | Number of perturbed structures to generate |
| `--cell-pert-fraction` | float | Fractional cell distortion magnitude (e.g. `0.03` = 3 %) |
| `--atom-pert-distance` | float | Maximum atomic displacement in Å |

**Optional flags**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--atom-pert-style` | str | `normal` | Displacement style: `normal` (Gaussian), `uniform` (uniform sphere), `const` (fixed magnitude) |
| `--atom-pert-prob` | float | `1.0` | Fraction of atoms displaced per frame (0–1) |
| `--output-format` | str | `extxyz` | Output format |
| `--output` | path | auto | Explicit output file path |

**JSON response fields**
```
status, message, structure_path, num_structures, num_atoms_per_structure
```

**Examples**
```bash
# 50 perturbed copies of Al FCC, 3% cell distortion, 0.1 Å atomic displacement
python skills/crystal_structure/crystal_structure_tools.py perturb-atoms Al_fcc.extxyz \
    --pert-num 50 --cell-pert-fraction 0.03 --atom-pert-distance 0.1

# Perturb only 80% of atoms, use uniform displacement style
python skills/crystal_structure/crystal_structure_tools.py perturb-atoms structure.extxyz \
    --pert-num 20 --cell-pert-fraction 0.02 --atom-pert-distance 0.15 \
    --atom-pert-style uniform --atom-pert-prob 0.8 --output perturbed.extxyz
```

---

### 4. `inspect-structure`

Read any ASE-readable structure file (single or multi-frame) and report metadata. Optionally export per-frame properties to text files.

**Positional arguments**
| Argument | Description |
|---|---|
| `input` | Path to structure file |

**Optional flags** (all default to `False`/omitted)
| Flag | Exported file | Description |
|---|---|---|
| `--export-volume` | `volumes.txt` | Per-frame volumes (Å³) |
| `--export-cell-parameters` | `cell_parameters.txt` | Per-frame *a, b, c, α, β, γ* |
| `--export-density` | `densities.txt` | Per-frame densities (g/cm³) |
| `--export-positions` | `positions.extxyz` | All frames as extxyz |
| `--export-forces` | `forces.txt` | Per-frame flattened force arrays (requires `forces` array key) |
| `--export-energy` | `energies.txt` | Per-frame energies in eV (auto-detects key) |
| `--export-stress` | `stresses.txt` | Per-frame Voigt stresses (requires `stress`/`virial` info key) |
| `--output-dir` | — | Directory for exported files (auto-generated if omitted) |

**JSON response fields** (always present)
```
status, message, structure_path, num_frames, chemical_formulas, num_atoms,
info_keys, array_keys
```
Additional fields appear when the corresponding `--export-*` flag is given (e.g. `volume_file`, `volume_summary`, `energy_file`, `energy_summary`, …).

**Examples**
```bash
# Basic metadata check
python skills/crystal_structure/crystal_structure_tools.py inspect-structure structures.extxyz

# Export volumes and energies to a specific directory
python skills/crystal_structure/crystal_structure_tools.py inspect-structure dataset.extxyz \
    --export-volume --export-energy --output-dir ./analysis
```

---

### 5. `transform-lattice`

Apply one or more lattice transformations (applied in order: scale → strain → rotation → custom matrix) to a structure. Atom positions are rescaled with the cell by default.

**Positional arguments**
| Argument | Description |
|---|---|
| `input` | Path to input structure file |

**Optional flags**
| Flag | Type | Description |
|---|---|---|
| `--scale` | float or JSON | Uniform scale factor, e.g. `0.97`, or anisotropic `[sx,sy,sz]` |
| `--strain` | JSON | Voigt 6-vector `[exx,eyy,ezz,eyz,exz,exy]` or 3×3 tensor |
| `--rotation` | JSON | ZYZ Euler angles `[α,β,γ]` in degrees, or 3×3 rotation matrix |
| `--transform-matrix` | JSON | Arbitrary 3×3 matrix; applied as `new_cell = M @ cell` |
| `--no-scale-atoms` | flag | Keep Cartesian positions fixed (do not rescale atoms) |
| `--output-format` | str | Output format (default: `extxyz`) |
| `--output` | path | Explicit output file path |

**JSON response fields**
```
status, message, structure_path, original_cell, transformed_cell,
operations_applied, scale_atoms, num_atoms
```

**Examples**
```bash
# Uniformly compress by 3%
python skills/crystal_structure/crystal_structure_tools.py transform-lattice structure.extxyz \
    --scale 0.97

# Apply a uniaxial strain of 2% along z
python skills/crystal_structure/crystal_structure_tools.py transform-lattice structure.extxyz \
    --strain "[0.0,0.0,0.02,0.0,0.0,0.0]"

# Anisotropic scale, output as VASP POSCAR
python skills/crystal_structure/crystal_structure_tools.py transform-lattice structure.extxyz \
    --scale "[1.0,1.0,0.95]" --output-format vasp --output POSCAR
```

---

### 6. `filter-by-entropy`

Select a maximally diverse subset of candidate structures using entropy-based (QUEST) filtering. Tries GPU (CUDA) first; falls back to CPU automatically. Outputs a multi-frame extxyz file of the selected structures.

**Positional arguments**
| Argument | Description |
|---|---|
| `iter_confs` | One or more candidate structure files (space-separated) |

**Optional flags**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--reference` | paths | `[]` | Reference structure files already in the dataset (excluded from selection but used to compute baseline entropy) |
| `--chunk-size` | int | `10` | Structures added per iteration |
| `--k` | int | `32` | Number of nearest neighbours for the QUEST descriptor |
| `--cutoff` | float | `5.0` | Descriptor cutoff radius in Å |
| `--batch-size` | int | `1000` | Batch size for entropy computation |
| `--h` | float | `0.015` | Bandwidth parameter *h* |
| `--max-sel` | int | `50` | Maximum structures to select |

**JSON response fields**
```
status, message, selected_atoms, entropy
```
`selected_atoms` is the absolute path to the output extxyz file.
`entropy` is a dict with per-iteration entropy values (`iter_00`, `iter_01`, …) and `num_confs`.

**Examples**
```bash
# Select up to 100 diverse structures from a pool
python skills/crystal_structure/crystal_structure_tools.py filter-by-entropy \
    candidates.extxyz --max-sel 100

# Select relative to an existing reference dataset
python skills/crystal_structure/crystal_structure_tools.py filter-by-entropy \
    new_structures.extxyz --reference existing_dataset.extxyz --max-sel 50

# Multiple candidate files, tighter bandwidth
python skills/crystal_structure/crystal_structure_tools.py filter-by-entropy \
    pool1.extxyz pool2.extxyz --h 0.01 --cutoff 6.0 --max-sel 200
```

---

## Reading JSON output

Always capture and parse stdout to get the result:

```bash
result=$(python skills/crystal_structure/crystal_structure_tools.py build-bulk-crystal Al fcc)
echo "$result" | python -c "import sys,json; d=json.load(sys.stdin); print(d['structure_path'])"
```

On error `"status"` is `"error"` and `"message"` contains the exception text. The exit code is also non-zero.