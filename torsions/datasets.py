from torch.utils.data import Dataset
from torch import FloatTensor
from os.path import join
from glob import glob
from pandas import read_csv
from numpy import zeros, stack, sin, cos, pi

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
        coords = stack([df.x, df.y, df.z], axis=1)

        return FloatTensor(sequence), FloatTensor(angles), FloatTensor(coords)