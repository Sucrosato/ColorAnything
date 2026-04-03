import torch
from torch import nn
import d2l
from cocoloader import cocoloader
from model.ColorAnything import ColorAnything
import os
from matplotlib import pyplot as plt
import argparse
from datetime import datetime
import sys
from pytorch_fid import fid_score
import cv2
import numpy as np
import logging

train_path = {'linux': '/public/Data/coco/images/train2017',
              'windows': 'E:/work/Code/ColorAnything/data/train_small'}
test_path = {'linux': '/public/Data/coco/images/test2017',
              'windows': 'E:/work/Code/ColorAnything/data/test_small'}


def calculate_colorfulness(image_bgr):
    """
    计算图像的 Colorfulness (Hasler & Suesstrunk, 2003)
    """
    # 确保数据类型为 float，防止计算负数时发生 uint8 溢出
    img = image_bgr.astype(np.float32)
    
    # 拆分 RGB 通道
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    # 1. 计算对立颜色通道
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    # 2. 计算均值 (Mean) 和 标准差 (Std)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    
    # 3. 组合指标
    std_root = np.sqrt(rg_std**2 + yb_std**2)
    mean_root = np.sqrt(rg_mean**2 + yb_mean**2)
    
    # 计算最终得分
    colorfulness = std_root + (0.3 * mean_root)
    
    return colorfulness

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

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = 1 #
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, platform, 
               devices=d2l.try_all_gpus(), animation=False):
    """Train a model with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    train_loss, test_colorfulness, test_fid = [], [], []
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], 0)
            timer.stop()
        # test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        with torch.no_grad():
            # 测试集fid分数 #
            batch_size = next(iter(test_iter))[0].shape[0]
            colorfulness_metric = d2l.Accumulator(2)
            for i, (x, _) in enumerate(test_iter):
                x = x.to('cuda')
                pred = net(x)
                synth = torch.concatenate((x, pred), axis=1)
                synth = torch.permute(synth, (0, 2, 3, 1))
                synth = torch.clamp(synth, 0.0, 1.0) * 255
                synth = synth.detach().cpu().numpy().astype('uint8')
                for j, img_lab in enumerate(synth):
                    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
                    colorfulness = calculate_colorfulness(img_bgr)
                    colorfulness_metric.add(1, colorfulness)

                    img_resize = cv2.resize(img_bgr, (299, 299), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f'../data/test_pred/{i * batch_size + j}.png', img_resize)

                    
                # synth = np.concatenate((x.detach().cpu().numpy(), pred.detach().cpu().numpy()), axis=1)
                # for j, pred in enumerate(synth):
                #     pred = np.transpose(pred, (1, 2, 0))
                #     pred = np.clip(pred, 0.0, 1.0)
                #     pred = (pred*255).astype('uint8') # to [0, 255]
                #     pred = cv2.resize(pred, (299, 299), interpolation=cv2.INTER_CUBIC)
                #     # pred = np.concatenate((grey_ori, pred), axis=2)
                #     pred = cv2.cvtColor(pred, cv2.COLOR_Lab2BGR)
                    
                #     cv2.imwrite(f'../data/test_pred/{i * batch_size + j}.png', pred)

            fid = fid_score.calculate_fid_given_paths(('../data/test_pred/', '../data/test_resized'), batch_size=batch_size, device='cuda:0', dims=2048)
            # #
            train_loss.append(metric[0] / metric[2])
            # train_accs.append(metric[1] / metric[3])
            test_colorfulness.append(colorfulness_metric[1] / colorfulness_metric[0])
            test_fid.append(fid)


        
    print(f'loss {metric[0] / metric[2]:.3f}\n'
          f'test colorfulness score {colorfulness_metric[1] / colorfulness_metric[0]}'
          f'test fid score {fid:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
    #add
    # return metric[0] / metric[2], metric[1] / metric[3], test_acc
    return train_loss, test_colorfulness, test_fid

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ColorAnything')
    
    parser.add_argument('--platform', type=str, required=True)
    parser.add_argument('--mode', type=str, default='lab')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=10)
    
    args = parser.parse_args()

    log_name = 'logs/' + datetime.now().strftime("%m%d-%H%M%S") + '.log'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('plt', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    logging.basicConfig(
        filename=log_name,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        print(datetime.now().strftime("%H:%M:%S"))

        platform = args.platform
        mode = args.mode

        if platform == 'linux':
            train_iter = cocoloader('/public/Data/coco/images/train2017', batch_size=16, mode=mode) #
            test_iter = cocoloader('/public/Data/coco/images/test2017', batch_size=16, mode=mode) #
        elif platform == 'windows':
            train_iter = cocoloader('E:/work/Code/ColorAnything/data/train_small', batch_size=4, mode=mode) #
            test_iter = cocoloader('E:/work/Code/ColorAnything/data/test_small', batch_size=4, mode=mode) #

        net = ColorAnything(platform=platform, mode=mode)
        net.to('cuda')

        start_epoch = args.start
        num_epochs = args.epoch
        lr = 5e-5
        if start_epoch > 0:
            net.load_state_dict(torch.load(f'./checkpoints/{mode}/{start_epoch}.pth'))

        optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
        # optimizer = get_optimizer(net, 5e-5, 5e-6, 0.001)

        train_loss, test_colorfulness, test_fids = train_ch13(net, train_iter, test_iter, nn.MSELoss(), optimizer, num_epochs, platform, animation=False)

        print(datetime.now().strftime("%H:%M:%S"))

        os.makedirs(f'./checkpoints/{mode}', exist_ok=True)
        torch.save(net.state_dict(), f'./checkpoints/{mode}/{start_epoch + num_epochs}.pth')


        epochs = range(1, num_epochs+1)
        with open(log_name, 'a+', encoding='utf-8') as f:
            f.write(str(train_loss))
            f.write(str(test_colorfulness))
            f.write(str(test_fids))
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(f'plt/Loss_{mode}_{num_epochs}.png')
        plt.close()
        
        plt.figure(figsize=(8, 5))
        # plt.plot(epochs, train_accs, 'bo-', label='Training Acc')
        plt.plot(epochs, test_colorfulness, 'ro-', label='Testing Colorfulness')
        plt.title('Colorfulness')
        plt.xlabel('Epochs')
        plt.ylabel('Colorfulness')
        plt.grid(True)
        plt.savefig(f'plt/Colorfulness_{mode}_{num_epochs}.png')
        plt.close()


        plt.figure(figsize=(8, 5))
        # plt.plot(epochs, train_accs, 'bo-', label='Training Acc')
        plt.plot(epochs, test_fids, 'ro-', label='Testing FID')
        plt.title('FID')
        plt.xlabel('Epochs')
        plt.ylabel('FID')
        plt.grid(True)
        plt.savefig(f'plt/FID_{mode}_{num_epochs}.png')
        plt.close()

        print('done.')
    
    except Exception as e:
        # exc_info=True 会自动把完整的报错堆栈写入日志
        logging.error("error caught", exc_info=True)
