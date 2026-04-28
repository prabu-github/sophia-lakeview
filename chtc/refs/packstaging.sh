#!/bin/bash

USER=$1
PROJECT=$2
MODELSLIST=$3

# extract project
echo "Extracting $PROJECT.tar.gz ..."
tar -xzf $PROJECT.tar.gz

# pack
echo "Packing ..."
cwd=$PWD/$PROJECT
python $cwd/chtc_pack_staging.py --chtc_user $USER --chtc_project_name $PROJECT --chtc_models_list $MODELSLIST

echo "Bye!"

exit
exit
