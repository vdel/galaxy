#!/bin/bash
OUTDIR=images_training_cropped 
mkdir $OUTDIR

for f in `ls images_training/`; do 
echo "Processing $f"
convert images_training/$f -crop 160x160+132+132 -resize 28x28 $OUTDIR/$f
done
