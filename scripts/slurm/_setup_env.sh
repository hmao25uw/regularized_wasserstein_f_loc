#!/bin/bash

# -----------------------------------------------------------------------------
# Common SLURM environment setup for NonparBayesCI.
#
# Why this exists:
#   Different clusters expose Julia/Gurobi in different ways (modules vs. user
#   installs). Hard-coding `module load gurobi` can break (as you've seen) when
#   the cluster doesn't provide a `gurobi` module.
#
# What this script does:
#   - If `module` is available, it *optionally* loads Julia and (optionally)
#     Gurobi modules.
#   - It NEVER hard-fails if a Gurobi module is missing.
#   - It DOES hard-fail if `julia` is not found after attempting to load it.
#
# Controls (environment variables):
#   JULIA_MODULE
#       If set, we run: module load "$JULIA_MODULE".
#       Example: export JULIA_MODULE=julia/1.10.4
#
#   NPEB_LOAD_GUROBI_MODULE
#       If set to 1, we attempt to load a Gurobi module.
#       Default: 0 (do not load any Gurobi module).
#
#   GUROBI_MODULE
#       If set (and NPEB_LOAD_GUROBI_MODULE=1), we run:
#         module load "$GUROBI_MODULE"
#       Otherwise we try `module load gurobi`.
#
# Notes:
#   - If you installed Gurobi yourself (no cluster module), do NOT set
#     NPEB_LOAD_GUROBI_MODULE=1; instead set GUROBI_HOME / LD_LIBRARY_PATH and
#     GRB_LICENSE_FILE in your sbatch script or shell profile.
# -----------------------------------------------------------------------------

set -euo pipefail

try_module_load() {
  local mod="$1"
  # `module` is often a shell function; `command -v` works for both functions & binaries.
  if ! command -v module >/dev/null 2>&1; then
    return 1
  fi
  # `set -e` would kill the script on a failed module load, so guard it.
  set +e
  module load "$mod" >/dev/null 2>&1
  local rc=$?
  set -e
  return $rc
}

# Attempt to load Julia if needed.
if ! command -v julia >/dev/null 2>&1; then
  if [[ -n "${JULIA_MODULE:-}" ]]; then
    if try_module_load "$JULIA_MODULE"; then
      echo "[env] loaded JULIA_MODULE=$JULIA_MODULE"
    else
      echo "[env] WARNING: failed to load JULIA_MODULE=$JULIA_MODULE" >&2
    fi
  else
    # Try a couple common module names; ignore failures.
    try_module_load "julia" || true
    try_module_load "julia/1.10" || true
  fi
fi

# Optionally attempt to load a Gurobi module.
if [[ "${NPEB_LOAD_GUROBI_MODULE:-0}" == "1" ]]; then
  if [[ -n "${GUROBI_MODULE:-}" ]]; then
    if try_module_load "$GUROBI_MODULE"; then
      echo "[env] loaded GUROBI_MODULE=$GUROBI_MODULE"
    else
      echo "[env] WARNING: failed to load GUROBI_MODULE=$GUROBI_MODULE (continuing)" >&2
    fi
  else
    if try_module_load "gurobi"; then
      echo "[env] loaded gurobi (default module)"
    else
      echo "[env] WARNING: module 'gurobi' not found (continuing without module)" >&2
    fi
  fi
fi

# Final sanity check: Julia must be runnable.
if ! command -v julia >/dev/null 2>&1; then
  echo "[env] ERROR: 'julia' not found on PATH." >&2
  echo "[env] Fix: load a Julia module in your sbatch script OR set JULIA_MODULE." >&2
  echo "[env] On Hyak Klone, try: module spider julia; then set JULIA_MODULE to the exact name." >&2
  exit 127
fi

echo "[env] julia at: $(command -v julia)"
julia --version || true

if [[ -n "${GRB_LICENSE_FILE:-}" ]]; then
  echo "[env] GRB_LICENSE_FILE=${GRB_LICENSE_FILE}"
fi
