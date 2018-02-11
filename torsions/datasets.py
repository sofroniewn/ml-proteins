from torch.utils.data import Dataset
from torch import FloatTensor
from os.path import join
from glob import glob
from pandas import read_csv
from numpy import zeros, nan_to_num, pad, sin, cos, concatenate

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
       'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


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
        a = df.aa.values
        sequence = zeros((len(a), 20))
        for i in range(len(a)):
            sequence[i, aa.index(a[i])] = 1

        angles = df[['chi', 'phi', 'psi']].values
        angles = nan_to_num(angles)
        angles = concatenate((sin(angles), cos(angles)), axis=1)
        length = angles.shape[0]
        if length < 200:
            offset = length % 2
            angles = pad(angles, (((int((200 - length) / 2), int((200 - length) / 2) + offset), (0, 0))),
                         mode='constant')
            sequence = pad(sequence, (((int((200 - length) / 2), int((200 - length) / 2) + offset), (0, 0))),
                           mode='constant')

        return FloatTensor(sequence.T), FloatTensor(angles.T)