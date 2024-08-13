import torch
from tqdm import tqdm

# torch geometric
try:
    import torch_geometric
except ModuleNotFoundError:
    print('Installing torch_geometric')
    import subprocess

    # Installing torch geometric packages with specific CUDA+PyTorch version.
    # See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html for details
    if torch.version.cuda is not None:
        TORCH = torch.__version__.split('+')[0]
        CUDA = 'cu' + torch.version.cuda.replace('.','')
        packages = ['torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv']
        for p in tqdm(packages, unit='pkg'):
            bashCommand = f"python3.8 -m pip install --no-index {p} -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        # bashCommand = f"python3.8 -m pip install torch_geometric"
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()

    else:
        # bashCommand = f"pip install torch_geometric"
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        # print(f'{output}, {error}')

        bashCommand = f"pip install  torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # bashCommand = f"python3.8 -m pip install torch_geometric"
    # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    # print(f'{output}, {error}')

    print(f'{output=}, {error=}')
    import torch_geometric

import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
