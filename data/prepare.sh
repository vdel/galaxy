#!/bin/bash
CROP=160
SIZE=60
OUTDIR=images_training_cropped_$CROP_$SIZE
mkdir $OUTDIR

let "PAD=(424-$CROP) / 2"

for f in `ls images_training/`; do 
echo "Processing $f"
convert images_training/$f -crop $CROPx$CROP+$PAD+$PAD -resize $SIZEx$SIZE $OUTDIR/$f
done
