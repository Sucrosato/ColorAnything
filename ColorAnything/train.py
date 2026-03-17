import torch
from torch import nn
import d2l
from cocoloader import cocoloader
from model.ColorAnything import ColorAnything


if __name__ == '__main__':
    net = ColorAnything()
    net.to('cuda')

    train_iter = cocoloader('/public/Data/coco/images/train2017', batch_size=4) #
    test_iter = cocoloader('/public/Data/coco/images/test2017', batch_size=4) #
    # train_iter = cocoloader(r'E:\work\Code\ColorAnything\data\train_small', batch_size=4) #
    # test_iter = cocoloader(r'E:\work\Code\ColorAnything\data\test_small', batch_size=4) #
    start_epoch = 0
    num_epochs = 10
    lr = 5e-5
    if start_epoch > 0:
        net.load_state_dict(torch.load(f'./checkpoints/{start_epoch}.pth'))
    trainer = torch.optim.AdamW(net.parameters(), lr=lr)
    d2l.train_ch13(net, train_iter, test_iter, nn.MSELoss(), trainer, num_epochs, animation=False)
    torch.save(net.state_dict(), f'./checkpoints/{start_epoch + num_epochs}.pth')