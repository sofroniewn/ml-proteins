import click
from torsions.datasets import PDBDataset
from torsions.model import LSTMaa
from torsions.main import run
from .common import success, status, error, warn
from torch.utils.data import DataLoader
from torch import cuda, load
from torch.autograd import Variable
from os.path import join, isdir, exists
from os import mkdir
from shutil import rmtree
from numpy import random
from pandas import DataFrame, read_csv

@click.argument('output', nargs=1, metavar='<output directory>', required=False, default=None)
@click.argument('model', nargs=1, metavar='<path to model>', required=True)
@click.argument('input', nargs=1, metavar='<input directory>', required=True)
@click.command('evaluate', short_help='evaluate model on input directory', options_metavar='<options>')

def evaluate_command(input, output, model):
    overwrite = True
    output = input if output is None else output

    status('setting up dataset from %s' % input)

    dataset = PDBDataset(input)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    status('loading model')
    if cuda.is_available():
        net = LSTMaa().cuda()
        net.load_state_dict(load(model))
    else:
        net = LSTMaa()
        net.load_state_dict(load(model, map_location=lambda storage, loc: storage))

    status('evaluating model')
    run(loader, net, output)
