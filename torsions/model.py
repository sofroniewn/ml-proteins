from torch import nn, cat
import torch.nn.functional as F
from torch import zeros
from torch.autograd import Variable
from torch import cuda
from torch.nn.utils.rnn import PackedSequence
from torch import transpose, mm, diag, clamp, sum
from numpy import inf

class LSTMaa(nn.Module):

    def __init__(self):
        super(LSTMaa, self).__init__()

        self.hidden_dim = 64

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(20, self.hidden_dim // 2, bidirectional=True, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2target = nn.Linear(self.hidden_dim, 12)

        # print(self.hidden)

    def init_hidden(self, minibatch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if cuda.is_available():
            return (Variable(zeros(2, minibatch_size, self.hidden_dim // 2)).cuda(),
                    Variable(zeros(2, minibatch_size, self.hidden_dim // 2)).cuda())
        else:
            return (Variable(zeros(2, minibatch_size, self.hidden_dim // 2)),
                    Variable(zeros(2, minibatch_size, self.hidden_dim // 2)))

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(sentence, self.hidden)

        if type(lstm_out) == PackedSequence:
            target = PackedSequence(self.hidden2target(lstm_out.data), lstm_out.batch_sizes)
        else:
            target = self.hidden2target(lstm_out)

        return target

from torch import atan2, stack, sin, cos, Tensor, cross, norm, mm
from numpy import pi
from torch.autograd import Variable

def position(A, B, C, bc, R, theta, phi):
    n = cross(B-A, C-B)
    n = n/norm(n)
    D = stack([R*cos(theta), R*sin(theta)*cos(phi), R*sin(theta)*sin(phi)])
    M = stack([(C-B)/bc, cross(n, C-B)/bc, n], dim=1)
    return mm(M,D).squeeze() + C

def reconstruct(ang, init):
    N_Ca = 1.458
    Ca_C = 1.525
    C_N = 1.329
    R = [C_N, N_Ca, Ca_C]
    bond_angles = stack([atan2(ang[:,0], ang[:,1]), atan2(ang[:,2], ang[:,3]), atan2(ang[:,4], ang[:,5])], dim=1).view(-1)
    torsion_angles = stack([atan2(ang[:,6], ang[:,7]), atan2(ang[:,8], ang[:,9]), atan2(ang[:,10], ang[:,11])], dim=1).view(-1)
    if cuda.is_available():
        pos = Variable(Tensor(len(bond_angles),3)).cuda()
    else:
        pos = Variable(Tensor(len(bond_angles),3))
    pos[0] = init[0]
    pos[1] = init[1]
    pos[2] = init[2]
    for ij in range(3, len(bond_angles)):
        pos[ij] = position(pos[ij-3], pos[ij-2], pos[ij-1], R[(ij-1)%3], R[ij%3], (pi-bond_angles[ij-1]), torsion_angles[ij-1])
    return bond_angles*180/pi, torsion_angles*180/pi, pos

def criterion_rmsd(outputs, labels):
    return (3**(0.5))*(outputs - labels).pow(2).mean().pow(0.5)

def pdist(x):
    x_norm = x.pow(2).sum(1).view(-1, 1)
    y_t = transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2 * mm(x, y_t)
    dist = dist - diag(dist.diag())
    dist = clamp(dist, 0.0, inf)
    #dist = dist.pow(0.5)
    dist[(dist != dist).detach()] = 0
    return dist


def criterion_drmsd(x, y):
#        return ((pdist(x) - pdist(y)).pow(2).mean()*len(x)/(len(x)-1)).pow(0.5)
    return (pdist(x) - pdist(y)).pow(2).mean()