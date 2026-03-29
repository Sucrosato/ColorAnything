import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from model.ColorAnything import ColorAnything


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ColorAnything')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=640)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--checkpoint', type=str)
    
    # parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    # parser.add_argument('--channel-only', dest='channel_only', action='store_true', help='only display the predicted channel')
    # parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='when channel-only, display channel in grayscale')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # model_configs = {
    #     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    # }
    
    # model = RGBAnything(**model_configs[args.encoder])
    model = ColorAnything()
    model.load_state_dict(torch.load(f'checkpoints/{args.checkpoint}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_img = cv2.imread(filename)
        
        img = model.infer_image(raw_img, args.input_size)
    
        split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([raw_img, split_region, img])
        
        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
        
        # if args.pred_only:
            # cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
        # else:
            # split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            # combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            # cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)