import cv2
import os
path = r'E:\work\Code\ColorAnything\data\test_small'
output_path = r'E:\work\Code\ColorAnything\data\test_resized'
# path = '/public/Data/coco/images/test2017'
filelist = os.listdir(path)
os.makedirs('./test_resized', exist_ok=True)
for file in filelist:
    img = cv2.imread(path+'/'+file)
    img = cv2.resize(img, (299, 299), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path+'/'+file, img)
