# hpc-and-latex

This repository contains an example deployment of publication-ready plots and tables
from a HPC workflow using LaTeX.

You are welcome to run this workflow locally, if you have a relatively modern Python
installation. You will need to install the dependencies in `requirements.txt` first, e.g.,

```bash
pip install -r requirements.txt
```

After which, edit the parameters in `makeparams.py` as desired, and run

```bash
./run_serial.sh
```

## On HPC

This workflow has been adapted to run on the Bridges-2 system at the Pittsburgh
Supercomputing Center (PSC).

To run on Bridges-2, you will need to have an allocation on the system. If you have
attended UCF's RCI workshop on Advanced HPC and LaTeX workflows, you are eligible to
use the allocation provided for participants. Contact your workshop instructor for
details.

### Accessing Bridges-2

Their user guide is here: https://www.psc.edu/resources/bridges-2/user-guide/.

Create a PSC password as described here: https://www.psc.edu/resources/bridges-2/user-guide#set-password.

Access the OnDemand Dashboard: https://ondemand.bridges2.psc.edu/, and log in with your PSC credentials.

Navigate to the "Clusters" tab, and select ">_Bridges-2 Shell Access".

#### Interactively

You will be placed into a login node initially, anything computationally intensive
should be done on a compute node. To request an interactive session on a compute node,
run:

```bash
srun --account=cis240124p --partition=RM-shared --cpus-per-task=4 --time=01:00:00 --pty bash
```

You will receive a meager 4 CPUs and 1 hour of time, but this should be sufficient.

From here, you *could* run the workflow directly, but that's no fun. Instead, you can run
some portions of your workflow automatically using Slurm batch jobs.

In order:

```bash
git clone https://github.com/benkeene/hpc-and-latex
cd hpc-and-latex
source scripts/create_env.sh
./run_serial.sh
```

This will run the complete experiment workflow and generate plots and figures.

#### Using batch submission

You can, instead of running everything as one main process (`./run_serial.sh`),
break the workflow into three pieces.

1) Experiment configuration, setting parameters and generating data (`makeparams.py`)
2) Batch submission of each experiment file
3) Collate results `python collate.py` and generate plots `python plots.py` and tables
python `tables.py`.

#### `run_parallel.sh`

On Unix systems (Linux, MacOS, WSL in Windows) you can leverage parallel computing tools
such as GNU Parallel, xargs, and others. Possible use cases include:

1) Starting a wholenode HPC job (~128 CPUs) to churn through small jobs in parallel
2) Performing small-scale experiments locally before running on HPC.