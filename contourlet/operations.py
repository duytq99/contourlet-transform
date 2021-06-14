import numpy as np
import torch
from contourlet import sampling
from torch.nn.functional import conv2d, pad


def lp_dec(img, h, g):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    height, w = img.shape[2], img.shape[3]
    low = conv2d(img, h, padding=4, groups=3)
    low = low[:, :,::2, ::2] 
    
    # low = qdown(img, m_down, torch.FloatTensor)
    # low_down = low[:,:,height//4:height*3//4,w//4:w*3//4]
    
    high = torch.zeros(img.shape)
    high[:, :,::2, ::2] = low 
    
    # high = qdown(low, m_up, torch.FloatTensor)
    
    high = conv2d(high.to(device), g.to(device), padding=3, groups=3)
    high = img - high
    
    return low, high

def lp_rec(high, low_band, h, g):
    high_ = high
    high = conv2d(high, h, padding=4, groups=3)
    high = high[:, :,::2, ::2]
    high = low_band - high
    
    shape = (low_band.shape[0], low_band.shape[1], low_band.shape[2]*2, low_band.shape[3]*2)
    img = torch.zeros(shape)
    img[:, :,::2, ::2] = high 
    img = conv2d(img, g, padding=3, groups=3)
    img = img + high_
    return img

def dfb_dec(img, h0, h1, name=None):
    h, w = img.shape[2], img.shape[3]
    if name=='haar':
        padding0 = (0,1)
        padding1 = (0,1)
    else:
        padding0 = 2
        padding1 = 2
    # level 1
    y0 = sampling.q_sampling(conv2d(img, h0, padding=padding0, groups=3), q_mode='q0', op_mode='down')
    y1 = sampling.q_sampling(conv2d(img, h1, padding=padding1, groups=3), q_mode='q0', op_mode='down')
    # level 2
    y00 = sampling.q_sampling(conv2d(y0, h0, padding=padding0, groups=3), q_mode='q1', op_mode='down')
    y01 = sampling.q_sampling(conv2d(y0, h1, padding=padding1, groups=3), q_mode='q1', op_mode='down')
    y10 = sampling.q_sampling(conv2d(y1, h0, padding=padding0, groups=3), q_mode='q1', op_mode='down')
    y11 = sampling.q_sampling(conv2d(y1, h1, padding=padding1, groups=3), q_mode='q1', op_mode='down')
    
    return torch.cat((y00, y01, y10, y11), dim=1)[:,:,h//4:h*3//4,w//4:w*3//4]

def dfb_rec(sub_bands, h0, h1, name=None):
    h, w = sub_bands.shape[2], sub_bands.shape[3]
    pad = torch.nn.ZeroPad2d((w//2,w//2,h//2,h//2))
    # print('pad: ', pad)
    if name=='haar':
        padding0 = (0,1)
        padding1 = (0,1)
    else:
        padding0 = 1
        padding1 = 3
    
    q0 = torch.tensor([[0.5, 0.5, 0],[-0.5, 0.5, 0]])
    q1 = torch.tensor([[0.5, -0.5, 0],[0.5, 0.5, 0]])
    
    y00 = sampling.q_sampling(pad(sub_bands[:,0:3]), q_mode='q1', op_mode='up')
    y01 = sampling.q_sampling(pad(sub_bands[:,3:6]), q_mode='q1', op_mode='up')
    y10 = sampling.q_sampling(pad(sub_bands[:,6:9]), q_mode='q1', op_mode='up')
    y11 = sampling.q_sampling(pad(sub_bands[:,9:12]), q_mode='q1', op_mode='up')
    
    y00 = conv2d(y00, h0, padding=padding0, groups=3)[:,:,0:h*2,0:w*2]
    y01 = conv2d(y01, h1, padding=padding1, groups=3)[:,:,0:h*2,0:w*2]
    y10 = conv2d(y10, h0, padding=padding0, groups=3)[:,:,0:h*2,0:w*2]
    y11 = conv2d(y11, h1, padding=padding1, groups=3)[:,:,0:h*2,0:w*2]
    # print('y00 y01 shape', y00.shape, y01.shape)
    
    
    y0 = y00 + y01
    y1 = y10 + y11
    
    y0 = sampling.q_sampling(y0, q_mode='q0', op_mode='up')
    y1 = sampling.q_sampling(y1, q_mode='q0', op_mode='up')
    
    y0 = conv2d(y0, h0, padding=padding0, groups=3)[:,:,0:h*2,0:w*2]
    y1 = conv2d(y1, h1, padding=padding1, groups=3)[:,:,0:h*2,0:w*2]
    # print('y0 y1 shape', y0.shape, y1.shape)
    return y0+y1