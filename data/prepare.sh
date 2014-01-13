#!/bin/bash
INDIR=images_training
CROP=120
SIZE=60

if [ $# -ge 1 ]; then
    INDIR=$1
fi
if [ $# -ge 2 ]; then
    SIZE=$2
fi
if [ $# -ge 3 ]; then
    CROP=$3
fi

OUTDIR=${INDIR}_cropped_${CROP}_${SIZE}
mkdir $OUTDIR

let "PAD=(424-${CROP}) / 2"

for f in `ls $INDIR`; do 
echo "Processing $f"
convert $INDIR/$f -crop ${CROP}x${CROP}+${PAD}+${PAD} -resize ${SIZE}x${SIZE} $OUTDIR/$f
done
