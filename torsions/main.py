from os.path import join
from torch.autograd import Variable
import torch
from numpy import pi, arctan2, where, minimum, nan_to_num
from pandas import DataFrame, read_csv
from glob import glob

def train(trainloader, net, criterion, optimizer, epoch, display):
    net.train()
    running_loss = 0.0
    count = 0.0
    results = DataFrame([])
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
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
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels).data[0]
        predict = outputs.squeeze(0).squeeze(0).data
        if torch.cuda.is_available():
            predict = predict.cpu().numpy().T
            valid = inputs.squeeze(0).squeeze(0).data.cpu().numpy().T.max(axis=1)
        else:
            predict = predict.numpy().T
            valid = inputs.squeeze(0).squeeze(0).data.numpy().T.max(axis=1)

        start = where(valid)[0][0]
        stop = where(valid)[0][-1]
        predict = predict[start:stop+1]

        if save:
            df = DataFrame({'chi': arctan2(predict[:, 0], predict[:, 3]) * 180 / pi,
                        'phi': arctan2(predict[:, 1], predict[:, 4]) * 180 / pi,
                        'psi': arctan2(predict[:, 2], predict[:, 5]) * 180 / pi})

            DataFrame(df).to_csv(join(output, 'predict_%05d.csv' % ind))

        total += labels.size(0)
        correct += loss
        r = {'epoch':[epoch+1], 'batch':[ind+1],'loss':[loss]}
        results = results.append(DataFrame(r), ignore_index=True)
        ind += 1

    print('Mean loss: %.2f %%' % (
        correct / total))
    return results

def run(loader, net, output):
    net.eval()
    ind = 0
    for data in loader:
        inputs, labels = data

        if torch.cuda.is_available():
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        predict = outputs.squeeze(0).squeeze(0).data
        if torch.cuda.is_available():
            predict = predict.cpu().numpy().T
            valid = inputs.squeeze(0).squeeze(0).data.cpu().numpy().T.max(axis=1)
        else:
            predict = predict.numpy().T
            valid = inputs.squeeze(0).squeeze(0).data.numpy().T.max(axis=1)

        start = where(valid)[0][0]
        stop = where(valid)[0][-1]
        predict = predict[start:stop+1]

        df = DataFrame({'chi': arctan2(predict[:, 0], predict[:, 3]) * 180 / pi,
                   'phi': arctan2(predict[:, 1], predict[:, 4]) * 180 / pi,
                   'psi': arctan2(predict[:, 2], predict[:, 5]) * 180 / pi})

        DataFrame(df).to_csv(join(output,'predict_%05d.csv' % ind))
        ind +=1

def summarize(input_dir, prediction_dir):
    input_files = sorted(glob(join(input_dir, '*.csv')))
    prediction_files = sorted(glob(join(prediction_dir, 'predict_*.csv')))

    results = DataFrame([])
    for idx in range(len(input_files)):
        inputs = read_csv(input_files[idx])
        inputs = inputs[:700]
        predictions = read_csv(prediction_files[idx])
        inputs.chi = nan_to_num(inputs.chi)
        results = results.append({'chi': MAE(inputs.chi, predictions.chi),
                                  'phi': MAE(inputs.phi, predictions.phi),
                                  'psi': MAE(inputs.psi, predictions.psi)}, ignore_index=True)
    results.to_csv(join(prediction_dir, 'results.csv'))
    return results


def MAE(x, y):
    return sum(minimum(abs(y - x), 360 - abs(y - x))) / len(x)