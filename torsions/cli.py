import click

settings = dict(help_option_names=['-h', '--help'])
from torsions.commands.train import train_command
from torsions.commands.eval import evaluate_command

@click.group(options_metavar='', subcommand_metavar='<command>', context_settings=settings)
def cli():
    """
    This is a tool for deep learning based torsion angle prediction.
    Check out the list of commands to see what you can do.
    """

cli.add_command(train_command)
cli.add_command(evaluate_command)
