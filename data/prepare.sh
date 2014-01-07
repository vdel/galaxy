#!/bin/bash
SIZE=60
OUTDIR=images_training_cropped_$SIZE
mkdir $OUTDIR

for f in `ls images_training/`; do 
echo "Processing $f"
convert images_training/$f -crop 160x160+132+132 -resize $SIZEx$SIZE $OUTDIR/$f
done
