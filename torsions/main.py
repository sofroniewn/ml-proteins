from os.path import join
from torch.autograd import Variable
import torch
from numpy import pi, arctan2, where, minimum, nan_to_num, stack, array
from numpy.linalg import norm
from pandas import DataFrame, read_csv
from glob import glob
from torsions.model import criterion_pos, reconstruct

def train(trainloader, net, criterion, optimizer, epoch, display):
    net.train()
    running_loss = 0.0
    count = 0.0
    results = DataFrame([])
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels, coords = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs, labels, coords = Variable(inputs).cuda(), Variable(labels).cuda(), Variable(coords).cuda()
        else:
            inputs, labels, coords = Variable(inputs), Variable(labels), Variable(coords)
        net.hidden = net.init_hidden(1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        bond_angles, torsion_angles, pos = reconstruct(outputs[0], coords[0,:3])
        loss = criterion(outputs, labels) + criterion_pos(pos, coords[0])
        loss.backward()
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

def validate(valloader, net, criterion, optimizer, epoch, save, output):
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
        loss = criterion(outputs, labels) + criterion_pos(pos, coords[0])

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
        inputs = read_csv(input_files[idx])
        predictions = read_csv(prediction_files[idx])
        coords_in = stack([inputs.x, inputs.y, inputs.z], axis=1)
        coords_pred = stack([predictions.x, predictions.y, predictions.z], axis=1)

        results = results.append({'bond_angle': MAE(inputs.bond_angle, predictions.bond_angle),
                                  'torsion_angle': MAE(inputs.torsion_angle, predictions.torsion_angle),
                                  'rmse': array([norm(x) for x in coords_in-coords_pred]).mean()}, ignore_index=True)
    results.to_csv(join(prediction_dir, 'results.csv'))
    return results


def MAE(x, y):
    return sum(minimum(abs(y - x), 360 - abs(y - x))) / len(x)