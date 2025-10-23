"""
This module generates individual parameter files for a
set of simulations.

Each parameter file is a YAML file containing parameter values.
"""

import os
import yaml
from itertools import product

params = {
    'name': 'testing',
    'widths': range(32, 65, 8),
    'depths': [2],
    'n_samples': range(50, 251, 50),
    'repetitions': range(1),
    'in_dim': 1,
    'out_dim': 1,
    'x_min': -1.0,
    'x_max': 1.0,
    'dx': 0.01,
    'num_epochs': 500,
    'make_slurm_script': True,
    'slurm_params': {
        'account': 'cis240124p',
        'partition': 'RM-shared',
        'cpus-per-task': 4,
        'time': '00:15:00',
    }
}


def generate_param_combinations(params):
    """
    Unpack a parameter dictionary, possibly containing
    lists, into a complete list of parameter dictionaries,
    each containing a single value for each parameter.

    Args:
        params: Dictionary of parameters, where each value,
                except for strings, is either a single value
                or an iterable of values.
    Returns:
        List of parameter dictionaries, each containing
        a single value for each parameter.
    """
    # Keys to exclude from cartesian product (preserved as-is)
    exclude_keys = {'slurm_params', 'make_slurm_script'}

    values = []
    keys = []

    for key in params:
        if key in exclude_keys:
            continue

        keys.append(key)
        val = params[key]
        # Each param (except for strings) should be iterable
        # if not iterable, make it a single-element list
        if hasattr(val, '__iter__') and not isinstance(val, str):
            values.append(list(val))
        else:
            values.append([val])

    all_params = []
    # *values unpacks the list of lists made earlier
    # product generates the Cartesian product of the lists
    for combo in product(*values):
        # combo is a tuple of parameter values
        # zip(keys, combo) pairs each key with its value
        # dict(...) makes a dictionary from the pairs
        param_dict = dict(zip(keys, combo))

        # Add back the excluded keys
        for key in exclude_keys:
            if key in params:
                param_dict[key] = params[key]

        all_params.append(param_dict)

    # Add total number of experiments and the
    # experiment index to each param dict
    total_experiments = len(all_params)
    for idx, param in enumerate(all_params, start=1):
        param['total_experiments'] = total_experiments
        param['experiment_index'] = idx

    return all_params


def write_param_file(param_dict):
    """
    Write an individual parameter dictionary to a YAML file.

    The file is saved to ./params/{name}/exp_{experiment_index}.yaml

    Args:
        param_dict: Dictionary containing parameters including 'name' and 'experiment_index'
    """
    # Create directory path
    dir_path = os.path.join('params', param_dict['name'])
    os.makedirs(dir_path, exist_ok=True)

    # Create file path
    file_name = f"exp_{param_dict['experiment_index']}.yaml"
    file_path = os.path.join(dir_path, file_name)

    # Write YAML file
    with open(file_path, 'w') as f:
        yaml.dump(param_dict, f, default_flow_style=False, sort_keys=False)

    return file_path


def write_slurm_script(param_dict):
    """
    Write a SLURM script for the given parameter dictionary.

    The file is saved to ./slurm/{name}/exp_{experiment_index}.sh

    Args:
        param_dict: Dictionary containing parameters including 'name', 'experiment_index', and 'slurm_params'
    """
    # Create directory path
    dir_path = os.path.join('slurm', param_dict['name'])
    os.makedirs(dir_path, exist_ok=True)

    # Create file path
    file_name = f"exp_{param_dict['experiment_index']}.sbatch"
    file_path = os.path.join(dir_path, file_name)

    # Get SLURM parameters
    slurm_params = param_dict.get('slurm_params', {})

    # Build SLURM script
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={param_dict['name']}_exp_{param_dict['experiment_index']}",
        f"#SBATCH --output=slurm/{param_dict['name']}/exp_{param_dict['experiment_index']}-%j.out",
        f"#SBATCH --error=slurm/{param_dict['name']}/exp_{param_dict['experiment_index']}-%j.err",
    ]

    # Add SLURM parameters from the dict
    if 'account' in slurm_params:
        script_lines.append(f"#SBATCH --account={slurm_params['account']}")
    if 'partition' in slurm_params:
        script_lines.append(f"#SBATCH --partition={slurm_params['partition']}")
    if 'cpus-per-task' in slurm_params:
        script_lines.append(f"#SBATCH --cpus-per-task={slurm_params['cpus-per-task']}")
    if 'time' in slurm_params:
        script_lines.append(f"#SBATCH --time={slurm_params['time']}")

    # Add blank line and job commands
    script_lines.extend([
        "",
        "# Load any required modules here",
        "# module load python",
        "",
        "# Run the experiment",
        f"python runner.py params/{param_dict['name']}/exp_{param_dict['experiment_index']}.yaml",
    ])

    # Write SLURM script
    with open(file_path, 'w') as f:
        f.write('\n'.join(script_lines) + '\n')

    return file_path


if __name__ == "__main__":
    all_params = generate_param_combinations(params)
    for param in all_params:
        file_path = write_param_file(param)
        print(f"Wrote parameter file: {file_path}")

        # Create SLURM script if slurm_params exists
        if 'slurm_params' in param and param['slurm_params']:
            slurm_path = write_slurm_script(param)
            print(f"Wrote SLURM script: {slurm_path}")