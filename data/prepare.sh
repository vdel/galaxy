#!/bin/bash
CROP=224
SIZE=224
OUTDIR=images_training_cropped_${CROP}_${SIZE}
mkdir $OUTDIR

let "PAD=(424-${CROP}) / 2"

for f in `ls images_training/`; do 
echo "Processing $f"
convert images_training/$f -crop ${CROP}x${CROP}+${PAD}+${PAD} -resize ${SIZE}x${SIZE} $OUTDIR/$f
done
