from os.path import join
from torch.autograd import Variable
import torch
from numpy import minimum, stack
from numpy.linalg import norm
from scipy.spatial.distance import pdist
from pandas import DataFrame, read_csv
from glob import glob
from torsions.model import reconstruct, criterion_rmsd, criterion_drmsd
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils import clip_grad_norm

def train(trainloader, net, criterion, optimizer, epoch, display, rmsd_loss, drmsd_loss):
    net.train()
    running_loss = 0.0
    count = 0.0
    results = DataFrame([])
    for i, data in enumerate(trainloader, 0):
        # get the inputs as packed sequences
        inputs, labels, coords = data

        net.hidden = net.init_hidden(trainloader.batch_size)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        if rmsd_loss:
            o, l_o = unpack(outputs, batch_first=True)
            c, l_c = unpack(coords, batch_first=True)
            loss_pos = 0
            for ij in range(trainloader.batch_size):
                bond_angles, torsion_angles, pos = reconstruct(o[ij, :l_o[ij]], c[ij, :3])
                loss_pos = loss_pos + criterion_rmsd(pos, c[ij, :l_c[ij]])
            loss_pos = loss_pos / trainloader.batch_size
            loss = loss_pos
        elif drmsd_loss:
            o, l_o = unpack(outputs, batch_first=True)
            c, l_c = unpack(coords, batch_first=True)
            loss_pos = 0
            for ij in range(trainloader.batch_size):
                bond_angles, torsion_angles, pos = reconstruct(o[ij, :l_o[ij]], c[ij, :3])
                loss_pos = loss_pos + criterion_drmsd(pos, c[ij, :l_c[ij]])
            loss_pos = loss_pos / trainloader.batch_size
            loss = loss_pos
        else:
            loss = criterion(outputs.data, labels.data)

        loss.backward()
        clip_grad_norm(net.parameters(), 0.5)

        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        count += 1
        r = {'epoch':[epoch+1],'batch':[i+1],'loss':[loss.data[0]]}
        results = results.append(DataFrame(r), ignore_index=True)
        if i % display == display-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / count))
            running_loss = 0.0
            count = 0.0
    return results

def validate(valloader, net, criterion, optimizer, epoch, save, output, rmsd_loss, drmsd_loss):
    net.eval()
    if save:
        torch.save(net.state_dict(), join(output, 'model.pth'))
        torch.save(optimizer.state_dict(), join(output, 'opt.pth'))
    correct = 0
    total = 0
    ind = 0
    results = DataFrame([])
    for data in valloader:
        inputs, labels, coords = data

        if torch.cuda.is_available():
            inputs, labels, coords = Variable(inputs).cuda(), Variable(labels).cuda(), Variable(coords).cuda()
        else:
            inputs, labels, coords = Variable(inputs), Variable(labels), Variable(coords)
        net.hidden = net.init_hidden(1)

        outputs = net(inputs)
        bond_angles, torsion_angles, pos = reconstruct(outputs[0], coords[0, :3])
        if rmsd_loss:
            loss = criterion_rmsd(pos, coords[0])
        elif drmsd_loss:
            loss = criterion_drmsd(pos, coords[0])
        else:
            loss = criterion(outputs, labels)

        if torch.cuda.is_available():
            bond_angles = bond_angles.data.cpu().numpy()
            torsion_angles = torsion_angles.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
        else:
            bond_angles = bond_angles.data.numpy()
            torsion_angles = torsion_angles.data.cpu().numpy()
            pos = pos.data.cpu().numpy()

        if save:
            df = DataFrame({'bond_angle': bond_angles,
                            'torsion_angle': torsion_angles,
                            'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]})

            DataFrame(df).to_csv(join(output, 'predict_%05d.csv' % ind))

        total += labels.size(0)
        correct += loss.data[0]
        r = {'epoch':[epoch+1], 'batch':[ind+1],'loss':[loss.data[0]]}
        results = results.append(DataFrame(r), ignore_index=True)
        ind += 1

    print('Mean loss: %.2f %%' % (
        correct / total))
    return results

def run(loader, net, output):
    net.eval()
    ind = 0
    for data in loader:
        inputs, labels, coords = data

        if torch.cuda.is_available():
            inputs, labels, coords = Variable(inputs).cuda(), Variable(labels).cuda(), Variable(coords).cuda()
        else:
            inputs, labels, coords = Variable(inputs), Variable(labels), Variable(coords)
        net.hidden = net.init_hidden(1)

        outputs = net(inputs)
        bond_angles, torsion_angles, pos = reconstruct(outputs[0], coords[0, :3])

        if torch.cuda.is_available():
            bond_angles = bond_angles.data.cpu().numpy()
            torsion_angles = torsion_angles.data.cpu().numpy()
            pos = pos.data.cpu().numpy()
        else:
            bond_angles = bond_angles.data.numpy()
            torsion_angles = torsion_angles.data.cpu().numpy()
            pos = pos.data.cpu().numpy()

        df = DataFrame({'bond_angle': bond_angles,
                            'torsion_angle': torsion_angles,
                            'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]})
        DataFrame(df).to_csv(join(output,'predict_%05d.csv' % ind))
        ind +=1

def summarize(input_dir, prediction_dir):
    input_files = sorted(glob(join(input_dir, '*.csv')))
    prediction_files = sorted(glob(join(prediction_dir, 'predict_*.csv')))

    results = DataFrame([])
    for idx in range(len(input_files)):
        print(idx)
        inputs = read_csv(input_files[idx])
        predictions = read_csv(prediction_files[idx])
        coords_in = stack([inputs.x, inputs.y, inputs.z], axis=1)
        coords_pred = stack([predictions.x, predictions.y, predictions.z], axis=1)

        results = results.append({'bond_angle': MAE(inputs.bond_angle[1:-1], predictions.bond_angle[1:-1]),
                                  'omega': MAE(inputs.torsion_angle[3::3], predictions.torsion_angle[3::3]),
                                  'psi': MAE(inputs.torsion_angle[4::3], predictions.torsion_angle[4::3]),
                                  'phi': MAE(inputs.torsion_angle[2:-1:3], predictions.torsion_angle[2:-1:3]),
                                  'RMSD': rmsd(coords_in, coords_pred),
                                  'dRMSD': dRMSD(coords_in, coords_pred),
                                  'length': len(inputs)/3}, ignore_index=True)
    results.to_csv(join(prediction_dir, 'results.csv'))
    return results


def MAE(x, y):
    return sum(minimum(abs(y - x), 360 - abs(y - x))) / len(x)

def dRMSD(x, y):
    return norm(pdist(x) - pdist(y))/((len(x)*(len(x)-1)/2)**(0.5))

def rmsd(outputs, labels):
    return (3**(0.5))*((outputs - labels)**(2)).mean()**(0.5)