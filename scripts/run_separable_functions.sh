#!/usr/bin/env bash
# Run the separable suite (5 functions) sequentially at D=10, 20, 30.
# 60s gap between functions within a dimension; 120s gap between dimensions.
set -euo pipefail

cd "$(dirname "$0")/.."

VARIANTS="lhs_init,OBL_init,chaotic_init,adaptive_tourney,clonal_expand,DE_mut,SHADE_mut,Levy_mut,CLONALG_hyper,gauss_cauchy_mut,adaptive_mut,encircling_mut,pher_guided_mut,multiparent_cx,pher_edge_cx,greedy_replace,archive_replace,crowding_replace,restart,immigrant"

FUNCS=(Sphere Rastrigin Schwefel SumOfDifferentPowers Step)
DIMS=(10 20 30)

for d in "${!DIMS[@]}"; do
    dim="${DIMS[$d]}"
    echo "############################################################"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting separable suite at D=${dim}"
    echo "############################################################"

    for i in "${!FUNCS[@]}"; do
        fn="${FUNCS[$i]}"
        echo "============================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting $fn (D=${dim})"
        echo "============================================================"

        python scripts/run_experiments.py --suite separable --exact --dim "$dim" --pop-size 100 --max-fes 200000 --n-runs 30 --workers 8 --functions "$fn" --tag "separable_D${dim}_${fn}_$(date '+%Y%m%d_%H%M%S')" --variants "$VARIANTS"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $fn (D=${dim})"

        if [ "$i" -lt $((${#FUNCS[@]} - 1)) ]; then
            echo "Sleeping 60s before next function..."
            sleep 60
        fi
    done

    if [ "$d" -lt $((${#DIMS[@]} - 1)) ]; then
        echo "Sleeping 60s before next dimension..."
        sleep 60
    fi
done

echo "All separable functions complete across D=10,20,30."
