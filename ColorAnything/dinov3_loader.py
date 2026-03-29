import torch
from torch import nn

def dinov3_pretrained(model_name='vits', platform='windows'):
    REPO_DIR = 'E:/work/Code/ColorAnything/ColorAnything/dinov3'
    if platform=='linux':
        REPO_DIR = '/public/home/lyzhao/dychen/ColorAnything-main/ColorAnything/dinov3'
    if model_name=='vits':
        model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='E:/work/Code/ColorAnything/ColorAnything/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
        if platform=='linux':
            model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights='/public/home/lyzhao/dychen/ColorAnything-main/ColorAnything/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
    old_conv = model.patch_embed.proj

    new_conv = nn.Conv2d(
        in_channels=1, 
        out_channels=old_conv.out_channels, 
        kernel_size=old_conv.kernel_size, 
        stride=old_conv.stride, 
        padding=old_conv.padding
    )
    with torch.no_grad():
        new_weight = old_conv.weight.sum(dim=1, keepdim=True) / 3.0
        new_conv.weight.copy_(new_weight)
        new_conv.bias.copy_(old_conv.bias)

    model.patch_embed.proj = new_conv

    return model
