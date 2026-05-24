#!/usr/bin/env bash
# Run the engineering suite (4 constrained design problems) sequentially,
# with a 60s gap between runs.
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANTS="lhs_init,OBL_init,chaotic_init,adaptive_tourney,clonal_expand,DE_mut,SHADE_mut,Levy_mut,CLONALG_hyper,gauss_cauchy_mut,adaptive_mut,encircling_mut,pher_guided_mut,multiparent_cx,pher_edge_cx,greedy_replace,archive_replace,crowding_replace,restart,immigrant"

FUNCS=(TensionCompressionSpring PressureVessel WeldedBeam SpeedReducer)

for i in "${!FUNCS[@]}"; do
    fn="${FUNCS[$i]}"
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $fn"
    echo "============================================================"

    python scripts/run_experiments.py --suite cec2020_realworld --exact --pop-size 100 --max-fes 200000 --n-runs 30 --workers 8 --functions "$fn" --tag "engineering_${fn}_$(date '+%Y%m%d_%H%M%S')" --variants "$VARIANTS"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $fn"

    if [ "$i" -lt $((${#FUNCS[@]} - 1)) ]; then
        echo "Sleeping 60s before next function..."
        sleep 60
    fi
done

echo "All engineering functions complete."
