#!/bin/bash
# Train only the runs that have no complete folder (summary.json) yet.
# Usage: bash run_missing.sh 1|2 [dry]
SCEN=$1
DRY=$2
ARCHS="${ARCHS:-fpn pan deeplabv3plus unetplusplus}"
if [ "$SCEN" = "1" ]; then
  CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
  PREFIX=""; EXTRA=""
else
  CATS="cable capsule carpet leather metal_nut pill screw tile zipper"
  PREFIX="scenario2_"; EXTRA="--scenario 2"
fi
for a in $ARCHS; do
  for c in $CATS; do
    if ls -d ~/thesis_runs/runs/${a}_${PREFIX}${c}_seed42_*/summary.json > /dev/null 2>&1; then
      echo "SKIP $a $c"; continue
    fi
    echo "RUN $a $c"
    [ "$DRY" = "dry" ] && continue
    python run.py seg $c --arch $a $EXTRA > ~/thesis_runs/logs/s${SCEN}_${a}_${c}_seed42.log 2>&1
  done
done
echo ALL DONE
