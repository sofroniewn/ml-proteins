from os.path import join
from torch.autograd import Variable
import torch
from pandas import DataFrame

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
            predict = predict.cpu().numpy()
        else:
            predict = predict.numpy()
        if save:
            DataFrame(predict).to_csv('predict_%05d.csv' % ind)

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
            predict = predict.cpu().numpy()
        else:
            predict = predict.numpy()

        DataFrame(predict).to_csv('predict_%05d.csv' % ind)
        ind +=1
