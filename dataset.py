import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os

class PtDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """
    # def __init__(self, data_dir, device, index_slice, transform=None):
        # self.device = device
    def __init__(self, data_dir, index_slice, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # f_trans = [f for f in os.listdir(data_dir)
        #            if f.endswith('trans_1d.pt')]
        f_trans = [f for f in os.listdir(data_dir)
                   if f.startswith('input')]
        f_trans.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        f_trans = f_trans[index_slice]

        # f_density = [f for f in os.listdir(data_dir)
        #              if f.endswith('density_1d.pt')]
        f_density = [f for f in os.listdir(data_dir)
                     if f.startswith('label')]
        f_density.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        f_density = f_density[index_slice]

        self.names = (f_trans, f_density)

    def __getitem__(self, index):
        # x = torch.load(os.path.join(self.data_dir, self.names[0][index]))
        # y = torch.load(os.path.join(self.data_dir, self.names[1][index]))
        x = np.load(os.path.join(self.data_dir, self.names[0][index]))
        y = np.load(os.path.join(self.data_dir, self.names[1][index]))
        # x = x.to(self.device)
        # y = y.to(self.device)

        # x = np.expand_dims(np.expand_dims(x, axis=1), axis=2)
        # y = np.expand_dims(np.expand_dims(y, axis=1), axis=2)

        data = {'input': x, 'label': y}

        if self.transform:
            data = self.transform(data)

        return data
    
    def __len__(self):
        return len(self.names[0])


class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""

  def __call__(self, data):
    input, label = data['input'], data['label']

    # Swap color axis because numpy image: H x W x C
    #                         torch image: C x H x W
    input = input.transpose((2, 0, 1))
    label = label.transpose((2, 0, 1))
    return {'input': torch.from_numpy(input),
            'label': torch.from_numpy(label)}
