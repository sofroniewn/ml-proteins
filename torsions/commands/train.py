import click
from torsions.datasets import PDBDataset, pad_packed_collate
from torsions.model import LSTMaa
from torsions.main import train, validate
from .common import success, status, error, warn
from torch import cuda, load
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from os.path import join, isdir, exists
from os import mkdir
from shutil import rmtree
from pandas import DataFrame, read_csv

@click.argument('output', nargs=1, metavar='<output directory>', required=False, default=None)
@click.argument('input', nargs=1, metavar='<input directory>', required=True)
@click.option('--resume', nargs=1, default=0, type=int, help='Epoch to resume from')
@click.option('--epochs', nargs=1, default=2, type=int, help='Number of epochs')
@click.option('--display', nargs=1, default=20, type=int, help='Number of train samples before displaying result')
@click.option('--save_epoch', nargs=1, default=None, type=int, help='Number of epochs before saving')
@click.option('--rmsd_loss', is_flag=True, default=False, help='Whether to use rmsd loss')
@click.option('--drmsd_loss', is_flag=True, default=False, help='Whether to use drmsd loss')
@click.option('--lr', nargs=1, default=0.01, type=float, help='Learning rate')
@click.command('train', short_help='train on input directory', options_metavar='<options>')

def train_command(input, output, epochs, display, lr, resume, save_epoch, rmsd_loss, drmsd_loss):
    overwrite = True


    if exists(join(output,'train.csv')):
        r = read_csv(join(output,'train.csv'))
    else:
        r = DataFrame([])
    if exists(join(output,'val.csv')):
        rV = read_csv(join(output,'val.csv'))
    else:
        rV = DataFrame([])

    status('setting up dataset from %s' % input)
    train_dataset = PDBDataset(join(input, 'train'))

    trainloader = DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=0, drop_last=True, collate_fn=pad_packed_collate)

    val_dataset = PDBDataset(join(input, 'val'))
    valloader = DataLoader(val_dataset, batch_size=1,
                                          shuffle=False, num_workers=0)

    status('loading model')
    if cuda.is_available():
        net = LSTMaa().cuda()
    else:
        net = LSTMaa()
    net.train()

    criterion = MSELoss(size_average=True)
    optimizer = Adam(net.parameters(), lr=lr)

    if resume is not 0:
        snapshot_name = 'model-%04d' % resume
        status('loading network %s' % snapshot_name)
        if cuda.is_available():
            net.load_state_dict(load(join(output, snapshot_name, 'model.pth')))
            optimizer.load_state_dict(load(join(output, snapshot_name, 'opt.pth')))
        else:
            net.load_state_dict(load(join(output, snapshot_name, 'model.pth'), map_location=lambda storage, loc: storage))
            optimizer.load_state_dict(load(join(output, snapshot_name, 'opt.pth'), map_location=lambda storage, loc: storage))

    status('starting training')
    for epoch in range(resume, resume+epochs):  # loop over the dataset multiple times
        results = train(trainloader, net, criterion, optimizer, epoch, display, rmsd_loss, drmsd_loss)
        r = r.append(results, ignore_index=True)
        r.to_csv(join(output,'train.csv'))

        #save out model every n epochs
        if save_epoch is not None:
            if epoch % save_epoch == save_epoch-1:
                snapshot_name = 'model-%04d' % (epoch + 1)
                status('saving network %s' % snapshot_name)
                save_path = join(output, snapshot_name)
                if isdir(save_path) and not overwrite:
                    error('directory already exists and overwrite is false')
                    return
                elif isdir(save_path) and overwrite:
                    rmtree(save_path)
                    mkdir(save_path)
                else:
                    mkdir(save_path)
                resultsV = validate(valloader, net, criterion, optimizer, epoch, True, save_path, rmsd_loss, drmsd_loss)
                rV = rV.append(resultsV, ignore_index=True)
                rV.to_csv(join(output,'val.csv'))

    status('finished training')
    snapshot_name = 'model-%04d' % (epoch + 1)
    status('saving network %s' % snapshot_name)
    save_path = join(output, snapshot_name)
    if isdir(save_path) and not overwrite:
        error('directory already exists and overwrite is false')
        return
    elif isdir(save_path) and overwrite:
        rmtree(save_path)
        mkdir(save_path)
    else:
        mkdir(save_path)
    resultsV = validate(valloader, net, criterion, optimizer, epoch, True, save_path, rmsd_loss)
    rV = rV.append(resultsV, ignore_index=True)
    rV.to_csv(join(output,'val.csv'))
