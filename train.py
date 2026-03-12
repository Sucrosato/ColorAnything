import torch
from torch import nn
from ColorAnything import d2l
from ColorAnything.cocoloader import cocoloader
from ColorAnything.model.ColorAnything import ColorAnything


if __name__ == '__main__':
    net = ColorAnything()
    net.to('cuda')

    train_iter = cocoloader('/public/Data/coco/images/train2017', batch_size=4) #
    test_iter = cocoloader('/public/Data/coco/images/test2017', batch_size=4) #
    start_epoch = 0
    num_epochs = 10
    lr = 5e-5
    if start_epoch > 0:
        net.load_state_dict(torch.load(f'./ColorAnything/checkpoints/{start_epoch}.pth'))
    trainer = torch.optim.AdamW(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, nn.MSELoss(), trainer, num_epochs)
    torch.save(net.state_dict(), f'./ColorAnything/checkpoints/{start_epoch + num_epochs}.pth')