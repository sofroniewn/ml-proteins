from torch.utils.data import Dataset
from torch import FloatTensor
from os.path import join
from glob import glob
from pandas import read_csv
from numpy import zeros, sin, cos, pi
from numpy import stack as stack_np
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch import cat, stack, zeros, cuda
from torch.autograd import Variable

aa = ['PRO', 'TYR', 'THR', 'VAL', 'PHE', 'ARG', 'GLY', 'CYS', 'ALA',
       'LEU', 'MET', 'ASP', 'GLN', 'SER', 'TRP', 'LYS', 'GLU', 'ASN',
       'ILE', 'HIS']

class PDBDataset(Dataset):
    """Dataset for parsed pdb files"""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the parsed pdb files.
        """
        self.root_dir = root_dir
        self.names = sorted(glob(join(self.root_dir, '*.csv')))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        df = read_csv(self.names[idx])
        sequence = zeros((len(df)//3, 20))
        angles = zeros((len(df)//3, 12))

        for i in range(len(df)//3):
            sequence[i, aa.index(df.aa[3*i])] = 1
            angles[i, 0] = sin(df.bond_angle[3*i]*pi/180)
            angles[i, 1] = cos(df.bond_angle[3*i]*pi/180)
            angles[i, 2] = sin(df.bond_angle[3*i+1]*pi/180)
            angles[i, 3] = cos(df.bond_angle[3*i+1]*pi/180)
            angles[i, 4] = sin(df.bond_angle[3*i+2]*pi/180)
            angles[i, 5] = cos(df.bond_angle[3*i+2]*pi/180)
            angles[i, 6] = sin(df.torsion_angle[3*i]*pi/180)
            angles[i, 7] = cos(df.torsion_angle[3*i]*pi/180)
            angles[i, 8] = sin(df.torsion_angle[3*i+1]*pi/180)
            angles[i, 9] = cos(df.torsion_angle[3*i+1]*pi/180)
            angles[i, 10] = sin(df.torsion_angle[3*i+2]*pi/180)
            angles[i, 11] = cos(df.torsion_angle[3*i+2]*pi/180)
        coords = stack_np([df.x, df.y, df.z], axis=1)

        return FloatTensor(sequence), FloatTensor(angles), FloatTensor(coords)


def stack_pack(var, lengths):
    max_len, n_feats = var[0].size()
    var = [cat((s, zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in var]
    var = stack(var, 0)
    var = Variable(var)
    if cuda.is_available():
        var = var.cuda()
    return pack(var, lengths, batch_first=True)


def pad_packed_collate(batch):
    """Puts data, and lengths into a packed_padded_sequence then returns
       the packed_padded_sequence and the labels. Set use_lengths to True
       to use this collate function.
       Args:
         batch: (list of tuples) [(aa, angles, coords)].
             aa is a FloatTensor
             angles is a FloatTensor
             coords is a FloatTensor
       Output:
         packed_aa: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         packed_angles: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
         packed_coords: (PackedSequence), see torch.nn.utils.rnn.pack_padded_sequence
    """

    if len(batch) == 1:
        aa, angles, coords = batch[0]
        lengths_aa = [aa.size(0)]
        lengths_coords = [coords.size(0)]
        aa.unsqueeze_(0)
        angles.unsqueeze_(0)
        coords.unsqueeze_(0)
    if len(batch) > 1:
        aa, angles, coords, lengths_aa, lengths_coords = zip(
            *[(a, b, c, a.size(0), c.size(0)) for (a, b, c) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
    packed_aa = stack_pack(aa, lengths_aa)
    packed_angles = stack_pack(angles, lengths_aa)
    packed_coords = stack_pack(coords, lengths_coords)
    return packed_aa, packed_angles, packed_coords