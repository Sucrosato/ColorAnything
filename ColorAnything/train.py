import torch
from torch import nn
import d2l
from cocoloader import cocoloader
from model.ColorAnything import ColorAnything
import os
from matplotlib import pyplot as plt

def get_optimizer(net, lr_base=5e-5, lr_backbone=5e-6, weight_decay=0.001):
    # backbone_params = net.pretrained.parameters()
    
    # head_params = net.depth_head.parameters()
    
    param_groups = [
        {
            'params': net.model[0].encoder.parameters(), 
            'lr': lr_backbone,       
            'weight_decay': weight_decay
        },
        {
            'params': net.model[0].decoder.parameters(), 
            'lr': lr_base,          
            'weight_decay': weight_decay
        }
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    
    return optimizer

if __name__ == '__main__':

    platform = 'windows'
    mode = 'lab'
    if platform == 'linux':
        train_iter = cocoloader('/public/Data/coco/images/train2017', batch_size=4, mode=mode) #
        test_iter = cocoloader('/public/Data/coco/images/test2017', batch_size=4, mode=mode) #
    elif platform == 'windows':
        train_iter = cocoloader(r'E:\work\Code\ColorAnything\data\train_small', batch_size=4, mode=mode) #
        test_iter = cocoloader(r'E:\work\Code\ColorAnything\data\test_small', batch_size=4, mode=mode) #

    net = ColorAnything(platform=platform, mode=mode)
    net.to('cuda')

    start_epoch = 0
    num_epochs = 5
    lr = 5e-5
    if start_epoch > 0:
        net.load_state_dict(torch.load(f'./checkpoints/{mode}/{start_epoch}.pth'))

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    # optimizer = get_optimizer(net, 5e-5, 5e-6, 0.001)

    train_loss, train_accs, test_accs = d2l.train_ch13(net, train_iter, test_iter, nn.MSELoss(), optimizer, num_epochs, animation=False)
    os.makedirs(f'./checkpoints/{mode}', exist_ok=True)
    torch.save(net.state_dict(), f'./checkpoints/{mode}/{start_epoch + num_epochs}.pth')


    epochs = range(1, num_epochs+1)
    
    print(train_loss, train_accs, test_accs)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'plt/Loss_{mode}_{num_epochs}.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, 'bo-', label='Training Acc')
    plt.plot(epochs, test_accs, 'ro-', label='Testing Acc')
    plt.title('Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.grid(True)
    plt.savefig(f'plt/Acc_{mode}_{num_epochs}.png')
    plt.close()
