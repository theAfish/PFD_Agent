# VASP Bohrium Submission Reference (bohr CLI)

See the `bohrium` skill for general `bohr` CLI usage (login, project ID, machine types, monitoring).
This file covers only **VASP-specific** parameters.

## VASP-specific environment variable

| Variable | Description | Example |
|---|---|---|
| `BOHRIUM_VASP_IMAGE` | Docker image URI for VASP | `registry.dp.tech/dptech/vasp:5.4.4` |

All other Bohrium parameters (`BOHRIUM_PROJECT_ID`, machine type) are general and not VASP-specific.

## VASP run command

```bash
source /opt/intel/oneapi/setvars.sh && mpirun -np <N_CORES> vasp_std > log 2> err
```

Match `N_CORES` to the machine's core count (e.g. 32 for `c32_m128_cpu`).

## VASP input/output files

| Calc type | Upload to `/input/` | Download from `/input/` |
|---|---|---|
| relaxation | POSCAR, INCAR, POTCAR, KPOINTS | CONTCAR, OUTCAR, vasprun.xml, OSZICAR, log, err |
| scf (nsoc) | POSCAR, INCAR, POTCAR, KPOINTS | CONTCAR, OUTCAR, vasprun.xml, CHGCAR, OSZICAR, log, err |
| scf (soc) | POSCAR, INCAR, POTCAR, KPOINTS | CONTCAR, OUTCAR, vasprun.xml, CHGCAR, WAVECAR, OSZICAR, log, err |
| nscf | POSCAR, INCAR, POTCAR, KPOINTS, CHGCAR [, WAVECAR] | OUTCAR, vasprun.xml, OSZICAR, log, err |

## Single job workflow

```bash
# 1. Submit — capture job ID
JOB_ID=$(bohr job submit \
  --project_id "$BOHRIUM_PROJECT_ID" \
  --name "vasp-relax-Al" \
  --machine_type "c32_m128_cpu" \
  --image "$BOHRIUM_VASP_IMAGE" \
  --disk_size 50 \
  --command "cd /input && source /opt/intel/oneapi/setvars.sh && mpirun -np 32 vasp_std" \
  | grep -oP '(?<="job_id":)\s*\d+' | tr -d ' ')

# 2. Upload input files
bohr job upload --job_id "$JOB_ID" --local_path <calc_dir>/ --remote_path /input/

# 3. Monitor (streams log until done)
bohr job log --job_id "$JOB_ID" --follow

# 4. Download results
bohr job download --job_id "$JOB_ID" --remote_path /input/ --local_path <calc_dir>/
```

## Batch submission (calc_dir_list)

Submit one job per `calc_dir`, record all job IDs, then poll for completion.

```bash
JOB_IDS=()

for CALC_DIR in <calc_dir_1> <calc_dir_2> ...; do
  NAME="vasp-$(basename $CALC_DIR)"
  JOB_ID=$(bohr job submit \
    --project_id "$BOHRIUM_PROJECT_ID" \
    --name "$NAME" \
    --machine_type "c32_m128_cpu" \
    --image "$BOHRIUM_VASP_IMAGE" \
    --disk_size 50 \
    --command "cd /input && source /opt/intel/oneapi/setvars.sh && mpirun -np 32 vasp_std > log 2> err" \
    | grep -oP '(?<="job_id":)\s*\d+' | tr -d ' ')
  bohr job upload --job_id "$JOB_ID" --local_path "$CALC_DIR"/ --remote_path /input/
  JOB_IDS+=("$JOB_ID:$CALC_DIR")
  echo "Submitted $NAME → job $JOB_ID"
done

# Poll until all done
while true; do
  ALL_DONE=true
  for ENTRY in "${JOB_IDS[@]}"; do
    JOB_ID="${ENTRY%%:*}"
    STATUS=$(timeout 60 bohr job status --job_id "$JOB_ID" 2>/dev/null | tr -d ' \n')
    [[ "$STATUS" != "finished" && "$STATUS" != "failed" && "$STATUS" != "cancelled" ]] && ALL_DONE=false
    echo "  job $JOB_ID: $STATUS"
  done
  $ALL_DONE && break
  sleep 120
done

# Download results for all completed jobs
for ENTRY in "${JOB_IDS[@]}"; do
  JOB_ID="${ENTRY%%:*}"
  CALC_DIR="${ENTRY#*:}"
  STATUS=$(timeout 60 bohr job status --job_id "$JOB_ID" 2>/dev/null | tr -d ' \n')
  if [[ "$STATUS" == "finished" ]]; then
    bohr job download --job_id "$JOB_ID" --remote_path /input/ --local_path "$CALC_DIR"/
    echo "Downloaded results for job $JOB_ID → $CALC_DIR"
  else
    echo "WARNING: job $JOB_ID ended with status $STATUS — check logs: bohr job log --job_id $JOB_ID"
  fi
done
```

## Handling failed jobs

```bash
# Inspect logs
bohr job log --job_id <JOB_ID> --tail 50

# Resubmit a failed job (same calc_dir, new job ID)
JOB_ID=$(bohr job submit \
  --project_id "$BOHRIUM_PROJECT_ID" \
  --name "vasp-retry-<name>" \
  --machine_type "c32_m128_cpu" \
  --image "$BOHRIUM_VASP_IMAGE" \
  --disk_size 50 \
  --command "cd /input && source /opt/intel/oneapi/setvars.sh && mpirun -np 32 vasp_std > log 2> err" \
  | grep -oP '(?<="job_id":)\s*\d+' | tr -d ' ')
bohr job upload --job_id "$JOB_ID" --local_path <calc_dir>/ --remote_path /input/
```
