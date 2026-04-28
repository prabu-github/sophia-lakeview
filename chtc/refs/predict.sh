#!/bin/bash

USER=$1
PROJECT=$2
MODEL=$3

# extract hytraits
echo "Extracting hytraits ..."
tar -xzf hytraits.tar.gz

# extract project
echo "Extracting $PROJECT.tar.gz ..."
tar -xzf $PROJECT.tar.gz

# train
echo "Predicting $MODEL ..."
python $PWD/$PROJECT/predict.py --pattern $MODEL

# package
if [ -d $PWD/$PROJECT/io_/predictions ]; then
    # Ship
    cp $PWD/$PROJECT/io_/predictions/$MODEL*.csv /staging/$USER/$PROJECT
fi

echo "Bye!"
 
exit
exit