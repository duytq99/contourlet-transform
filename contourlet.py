import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import torch
from skimage import metrics
from contourlet import filters, operations, sampling, transform

if __name__ == '__main__':
    device = 'cpu'
    img = cv2.imread(r'E:\Project_DL\NCKH_GUI_app\img_test\lol_dataset\eval15\high\1.png')
    # img = cv2.imread(r'image/lena.png')
    h, w = img.shape[0], img.shape[1]
    img = np.transpose(img / 255.0, (2, 0,  1)).astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(0).expand(1,3,h,w)


    img_ = transform.contourlet_transform(img)
    

    img_ = torch.clamp(img_, 0, 1)
    plt.imshow(np.transpose(img_[0].numpy(), (1,2,0)))
    plt.show()  

    
    print(metrics.peak_signal_noise_ratio(img.numpy().astype(np.float32), img_.clip(0.0 ,1.0).numpy().astype(np.float32)))
    mse = (img - img_.clip(0, 1.0)) ** 2
    print(10 * np.log10(1.0**2 / mse.mean()))