import torch
from torch import nn

def dinov2_pretrained(model_name='vits'):
    if model_name=='vits':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
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
