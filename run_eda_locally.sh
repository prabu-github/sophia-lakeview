#!/bin/bash

for i in {0..59}; do
    python trait.py --eda_config_idx $i
done

