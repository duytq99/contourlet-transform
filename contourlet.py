import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import torch
from skimage import metrics
from contourlet import filters, operations, sampling, transform

if __name__ == '__main__':
    device = 'cpu'
    img = cv2.imread('image/547 - Copy.png', 1)
    # img = np.expand_dims(img, -1)
    h, w = img.shape[0], img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img/255.0, (2, 0, 1)).astype(np.float32)
    img = torch.from_numpy(img).unsqueeze(0).expand(1,3,h,w)
    
    print(img.shape)
    
    low_band, high_band = transform.contourlet_decompose(img)
    # low_low_band, sub_band = transform.contourlet_decompose(low_band)
    
    # img__ = transform.contourlet_recompose(low_low_band, sub_band)
    img_ = transform.contourlet_recompose(low_band, high_band)
    
    print(img_.shape)

    
    
    plt.imshow(np.transpose(img_[0].numpy(), (1,2,0)))
    plt.show()
    
    print(metrics.peak_signal_noise_ratio(img.numpy(), img_.numpy()))
    