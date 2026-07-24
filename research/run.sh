if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

# # bash mrs_run.sh
# bash nihss_run_soop_outcome_experiments_brainiac.sh

bash raw_soop_kfold_mrs.sh&
bash raw_soop_kfold_nihss.sh&


bash synthetic_kfold_mrs.sh&
bash synthetic_kfold_nihss.sh&
