#!/bin/bash

declare -i cutoff=4
declare -i do_perms=1

python3 ave_perms.py $cutoff
python3 invert_PCA.py $cutoff 2022 04 13 $do_perms
