import torch

REPO_DIR = 'E:\work\Code\ColorAnything\ColorAnything\dinov3'

# DINOv3 ViT models pretrained on web images
dinov3_vits16 = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='facebook/dinov3-vits16-pretrain-lvd1689m')
