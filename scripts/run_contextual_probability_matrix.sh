#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 FD002_CHECKPOINT FD004_CHECKPOINT METROPT3_CHECKPOINT OUTPUT_ROOT" >&2
  echo "dry-run only; this script never launches training" >&2
  exit 2
fi

declare -A checkpoints=(
  [fd002]="$(realpath "$1")"
  [fd004]="$(realpath "$2")"
  [metropt3]="$(realpath "$3")"
)
output_root="$(realpath -m "$4")"
datasets=(fd002 fd004 metropt3)
seeds=(2026 2027 2028)
modes=(C0 C1 C3)
missing=()

for dataset in "${datasets[@]}"; do
  checkpoint="${checkpoints[$dataset]}"
  if [[ ! -f "$checkpoint" ]]; then
    missing+=("$checkpoint")
  fi
done

if (( ${#missing[@]} > 0 )); then
  printf 'BLOCKED: missing audited checkpoint candidates:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 3
fi

for dataset in "${datasets[@]}"; do
  checkpoint="${checkpoints[$dataset]}"
  for seed in "${seeds[@]}"; do
    for mode in "${modes[@]}"; do
      run_id="${dataset}_${mode,,}_seed${seed}"
      printf 'python scripts/run_contextual_probability.py --dataset %s --seed %s --checkpoint %s --mode %s --output-dir %s --run-id %s --epochs <PREREGISTERED_EPOCHS> --device cuda --evaluate-test\n' \
        "$dataset" "$seed" "$checkpoint" "$mode" "$output_root" "$run_id"
    done
  done
done
for dataset in fd002 fd004; do
  checkpoint="${checkpoints[$dataset]}"
  for seed in "${seeds[@]}"; do
    run_id="${dataset}_cpoint_seed${seed}"
    printf 'python scripts/run_contextual_probability.py --dataset %s --seed %s --checkpoint %s --mode Cpoint --training-target raw --output-dir %s --run-id %s --epochs <PREREGISTERED_EPOCHS> --device cuda --evaluate-test\n' \
      "$dataset" "$seed" "$checkpoint" "$output_root" "$run_id"
  done
done

echo "DRY-RUN ONLY: no experiment was started. Replace the preregistered epoch token only after launch-gate approval."
