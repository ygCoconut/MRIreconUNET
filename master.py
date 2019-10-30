"""
Idea behind this file.

1) The specs are specific for every run and they are set
in config.txt:
EPOCHS = ..
LR = ..
MODELDIRKEYWORD = ..
ARCH = .. (ARCHITECTURE)
WEIGHT DECAY = ..
..

2) You call in the command prompt:
python masterfile.py config.txt

3) This masterfile is then executed and calls in the commandprompt
python runall.py EPOCHS LR WEIGHT_DECAY etc..

3) runall.py is run with the above mentioned specs
as parameter set for argv.

GOAL: This procedure allows to launch runs from
the command line, especially from the cluster.
Furthermore, this allows to launch models in
parallel, as it becomes possible to have many
pending jobs at the same time whcih access different
text files instead of accessing different versions
of runall.py.

"""
import pandas as pd
import os
import sys

CONFIG = sys.argv[1]
specs = pd.read_csv(str(CONFIG))
specs['Parameters'] = specs['Parameters'].str.split(" = ")


RUNALL = specs['Parameters'][0][1]
EPOCHS = specs['Parameters'][1][1]
LR = specs['Parameters'][2][1]
WEIGHT_DECAY = specs['Parameters'][3][1]
BATCH_SIZE = specs['Parameters'][4][1]
MODELDIRKEYWORD = specs['Parameters'][5][1]
ARCH = specs['Parameters'][6][1]

output = str("python "
            + RUNALL + " "
            + EPOCHS + " "
            + LR + " "
            + WEIGHT_DECAY + " "
            + BATCH_SIZE + " "
            + CONFIG + " " #allows to later copy CONFIG to results (c.f. runall)
            # + MODELDIRKEYWORD + " "
            )

os.system(output)
