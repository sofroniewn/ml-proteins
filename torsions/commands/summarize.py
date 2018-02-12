import click
from torsions.main import summarize
from .common import success, status, error, warn


@click.argument('output', nargs=1, metavar='<path to model predictions>', required=True)
@click.argument('input', nargs=1, metavar='<input directory>', required=True)
@click.command('summarize', short_help='summarize angle error results', options_metavar='<options>')

def summarize_command(input, output):
    overwrite = True

    status('loading input data from from %s' % input)
    status('loading predictions data from from %s' % output)

    summarize(input, output)
