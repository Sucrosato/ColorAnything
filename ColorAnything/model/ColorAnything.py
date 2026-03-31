import torch
from torch import nn
import torch.nn.functional as F
from .encoder_decoder import build_depther
from dinov3_loader import dinov3_pretrained
from torchvision.transforms import Compose
from .util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import numpy as np

class ColorAnything(nn.Module):
    def __init__(self, mode='rgb', platform='windows'):
        super(ColorAnything, self).__init__()
        self.mode = mode

        modelist = {'rgb': 3, 'lab': 2}

        self.model = build_depther(
            dinov3_pretrained(platform=platform),
            [2, 5, 8, 11],
            modelist[mode],
            head_type='sdt'
        )
        if mode=='rgb':
            self.mean = np.array([0.485, 0.456, 0.406])
            self.std = np.array([0.229, 0.224, 0.225])
    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def infer_image(self, raw_image, input_size=224, compare=False): #
        grey, (h, w) = self.image2tensor(raw_image, input_size)
        if self.mode=='rgb':
            
            pred = self.forward(grey.unsqueeze(0))
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)# restore size
            pred = pred.cpu().numpy() * self.std[None, :, None, None] + self.mean[None, :, None , None]  # denormalize
            pred = np.clip(pred, 0.0, 1.0) # to [0, 1]
            pred = (pred * 255.0).astype('uint8') # to [0, 255]
            pred = np.squeeze(pred, axis=0)
            pred = np.transpose(pred, (1, 2, 0))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        elif self.mode=='lab':
            grey_ori = cv2.cvtColor(raw_image, cv2.COLOR_BGR2Lab)[:, :, 0:1]
            pred = self.forward(grey.unsqueeze(0)).squeeze(0).cpu().numpy()
            # pred = np.concatenate((grey.cpu(), pred.squeeze(0).cpu()), axis=0) 
            pred = np.transpose(pred, (1, 2, 0))
            pred = np.clip(pred, 0.0, 1.0)
            pred = (pred*255).astype('uint8') # to [0, 255]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_CUBIC)
            pred = np.concatenate((grey_ori, pred), axis=2)
            pred = cv2.cvtColor(pred, cv2.COLOR_Lab2BGR)



        return pred

    def image2tensor(self, raw_image, input_size=224):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=16,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            # NormalizeImage(mean=[0.485, 0.406], std=[0.229, 0.225]),
            # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            PrepareForNet(),
        ])
        
        h, w = raw_image.shape[:2]

        if self.mode == 'rgb':
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            grey = np.sum(img, axis=-1) / 3.0
            grey = torch.from_numpy(grey).unsqueeze(0)
        elif self.mode == 'lab':
            img = cv2.cvtColor(raw_image, cv2.COLOR_BGR2Lab)
            img = img.astype(np.float32) / 255.0
            img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            grey = img[:, :, 0]
            grey = torch.from_numpy(grey).unsqueeze(0)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        grey = grey.to(DEVICE)
        
        return grey, (h, w)
