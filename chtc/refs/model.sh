#!/bin/bash

USER=$1
PROJECT=$2
NOUTERS=$3
NINNERS=$4
MODEL=$5

# extract hytraits
echo "Extracting hytraits ..."
tar -xzf hytraits.tar.gz

# extract project
echo "Extracting $PROJECT.tar.gz ..."
tar -xzf $PROJECT.tar.gz

# train
echo "Training $MODEL ..."
python $PWD/$PROJECT/model.py --n_outers $NOUTERS --n_inners $NINNERS --pattern $MODEL

# package
if [ -d $PWD/$PROJECT/io_/model ]; then
    echo "Packaging $MODEL.tar.gz"
    PACKDIR=$PWD/$PROJECT/io_/$MODEL
    mkdir $PACKDIR
    mv $PWD/$PROJECT/io_/model $PACKDIR
    mv $PWD/$PROJECT/io_/deploy $PACKDIR
    tar -czf $PWD/$PROJECT/io_/$MODEL.tar.gz -C $PACKDIR .

    # Ship
    echo "Shipping $MODEL.tar.gz"
    cp $PWD/$PROJECT/io_/$MODEL.tar.gz /staging/$USER/$PROJECT
fi

echo "Bye!"
 
exit
exit