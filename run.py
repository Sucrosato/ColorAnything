import argparse
import cv2
import glob
import numpy as np
import os
import torch

from ColorAnything.model.ColorAnything import ColorAnything


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ColorAnything')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--output-size', type=int, default=0)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--compare', action='store_true')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model = ColorAnything(mode=args.mode)
    model.load_state_dict(torch.load(f'checkpoints/{args.mode}/{args.checkpoint}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    # print(filenames)
    os.makedirs(args.outdir, exist_ok=True)
    
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_img = cv2.imread(filename)
        img = model.infer_image(raw_img, args.input_size)
        if args.compare:
        
            split_region = np.ones((raw_img.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_img, split_region, img])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
        else:
            if args.output_size != 0:
                img = cv2.resize(img, (args.output_size, args.output_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), img)