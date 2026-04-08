import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class COCOColorInversionDataset(Dataset):
    def __init__(self, img_dir, input_size=224, mode='rgb'):
        if img_dir.endswith('.txt'):
            pass #
        else:
            self.img_dir = img_dir
            self.img_names = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

        self.input_size = input_size
        if mode=='rgb':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
        self.mode = mode

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_names[idx])
        raw_img = cv2.imread(img_path)
        resized_img = cv2.resize(raw_img, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC) # to input size

        if self.mode == 'rgb':
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # to rgb
            img = img.astype(np.float32) / 255.0 # to [0, 1]
            img = (img - self.mean) / self.std # normalize

            grey = np.sum(img, axis=-1) / 3.0
            grey = torch.from_numpy(grey).float().unsqueeze(dim=0)
            label = torch.from_numpy(img).permute(2, 0, 1).float()

        elif self.mode == 'lab':
            img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2Lab)
            img = img.astype(np.float32) / 255.0

            grey = torch.from_numpy(img[:, :, 0]).float().unsqueeze(dim=0)
            label = torch.from_numpy(img[:, :, (1, 2)]).permute(2, 0, 1).float()

        return grey, label, resized_img
    
def cocoloader(img_dir, batch_size=16, shuffle=True, num_workers=4, input_size=224, mode='rgb'):
    dataset = COCOColorInversionDataset(img_dir, input_size=input_size, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader