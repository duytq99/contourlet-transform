import numpy as np
import torch
from contourlet import sampling
from torch.nn.functional import conv2d, pad, conv1d


def lp_dec(img, h, g):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    height, w = img.shape[2], img.shape[3]
    padding_per = torch.nn.ReflectionPad2d((4,4,4,4))
    low = conv2d(padding_per(img), h, padding=0, groups=3)
    low = low[:, :,::2, ::2]
    
    high = torch.zeros(img.shape)
    high[:, :,::2, ::2] = low
    
    padding_per = torch.nn.ReflectionPad2d((3,3,3,3))
    high = conv2d(padding_per(high).to(device), g.to(device), padding=0, groups=3)
    high = img - high
    
    return low, high

def lp_dec_conv1d(img, h, g):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    h = np.array([.037828455506995, -.023849465019380, -.11062440441842, .37740285561265])
    h = np.hstack((h, .85269867900940, h[::-1]))
    h = np.expand_dims(h, 0).astype(np.float32)
    h = torch.from_numpy(h).unsqueeze(0)
    h = h.expand(3, 1, 1, 9).to(device)
    
    g = np.array([-.064538882628938, -.040689417609558, .41809227322221])
    g = np.hstack((g, .78848561640566, g[::-1]))
    g = np.expand_dims(g, 0).astype(np.float32)
    g = torch.from_numpy(g).unsqueeze(0)
    g = g.expand(3, 1, 1, 7).to(device)
    
    height, w = img.shape[2], img.shape[3]
    padding_per = torch.nn.ReflectionPad2d((4,4,4,4))
    low = conv1d(conv1d(padding_per(img), h, padding=0, groups=3), h.permute(0,1,3,2), padding=0, groups=3)
    low = low[:, :,::2, ::2]

    high = torch.zeros((img.shape[0], img.shape[1], height, w))
    high[:, :,::2, ::2] = low
    
    padding_per = torch.nn.ReflectionPad2d((3,3,3,3))
    high = conv1d(conv1d(padding_per(high), g, padding=0, groups=3), g.permute(0,1,3,2), padding=0, groups=3)
    high = img - high
    
    return low, high

def lp_rec(low_band, high, h, g):
    high_ = high
    padding_per = torch.nn.ReflectionPad2d((4,4,4,4))
    high = conv2d(padding_per(high), h, padding=0, groups=3)
    high = high[:, :,::2, ::2]
    high = low_band - high
    
    shape = (low_band.shape[0], low_band.shape[1], low_band.shape[2]*2, low_band.shape[3]*2)
    img = torch.zeros(shape)
    img[:, :,::2, ::2] = high
    padding_per = torch.nn.ReflectionPad2d((3,3,3,3))
    img = conv2d(padding_per(img), g, padding=0, groups=3)
    img = img + high_
    return img

def lp_rec_conv1d(low_band, high, h, g):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    h = np.array([.037828455506995, -.023849465019380, -.11062440441842, .37740285561265])
    h = np.hstack((h, .85269867900940, h[::-1]))
    h = np.expand_dims(h, 0).astype(np.float32)
    h = torch.from_numpy(h).unsqueeze(0)
    h = h.expand(3, 1, 1, 9).to(device)
    
    g = np.array([-.064538882628938, -.040689417609558, .41809227322221])
    g = np.hstack((g, .78848561640566, g[::-1]))
    g = np.expand_dims(g, 0).astype(np.float32)
    g = torch.from_numpy(g).unsqueeze(0)
    g = g.expand(3, 1, 1, 7).to(device)
    
    high_ = high
    padding_per = torch.nn.ReflectionPad2d((4,4,4,4))
    high =  conv1d(conv1d(padding_per(high), h, padding=0, groups=3), h.permute(0,1,3,2), padding=0, groups=3)
    high = high[:, :,::2, ::2]
    high = low_band - high
    
    shape = (low_band.shape[0], low_band.shape[1], low_band.shape[2]*2, low_band.shape[3]*2)
    img = torch.zeros(shape)
    img[:, :,::2, ::2] = high
    padding_per = torch.nn.ReflectionPad2d((3,3,3,3))
    img = conv1d(conv1d(padding_per(img), g, padding=0, groups=3), g.permute(0,1,3,2), padding=0, groups=3)
    img = img + high_
    return img

def dfb_dec(img, h0, h1, name=None):
    h, w = img.shape[2], img.shape[3]
    if name=='haar':
        padding0 = (0,1)
        padding1 = (0,1)
    else:
        padding_per = torch.nn.ReflectionPad2d((2,2,2,2))

    # level 1
    img = padding_per(img)
    y0 = sampling.q_sampling(conv2d(img, h0, padding=0, groups=3), q_mode='q0', op_mode='down')
    y1 = sampling.q_sampling(conv2d(img, h1, padding=0, groups=3), q_mode='q0', op_mode='down')
    # level 2
    y00 = sampling.q_sampling(conv2d(y0, h0, padding=0, groups=3), q_mode='q1', op_mode='down')
    y01 = sampling.q_sampling(conv2d(y0, h1, padding=0, groups=3), q_mode='q1', op_mode='down')
    y10 = sampling.q_sampling(conv2d(y1, h0, padding=0, groups=3), q_mode='q1', op_mode='down')
    y11 = sampling.q_sampling(conv2d(y1, h1, padding=0, groups=3), q_mode='q1', op_mode='down')
    
    return torch.cat((y00, y01, y10, y11), dim=1)[:,:,h//4:h*3//4,w//4:w*3//4]

def dfb_rec(sub_bands, h0, h1, name=None):
    h, w = sub_bands.shape[2], sub_bands.shape[3]
    pad = torch.nn.ZeroPad2d((w//2,w//2,h//2,h//2))
    padding_per = torch.nn.ReflectionPad2d((2,2,2,2))
    # print('pad: ', pad)
    if name=='haar':
        padding0 = (0,1)
        padding1 = (0,1)
    else:
        padding_per1 = torch.nn.ReflectionPad2d((1,1,1,1))
        padding_per3 = torch.nn.ReflectionPad2d((3,3,3,3))
    
    q0 = torch.tensor([[0.5, 0.5, 0],[-0.5, 0.5, 0]])
    q1 = torch.tensor([[0.5, -0.5, 0],[0.5, 0.5, 0]])
    
    y00 = sampling.q_sampling(pad(sub_bands[:,0:3]), q_mode='q1', op_mode='up')
    y01 = sampling.q_sampling(pad(sub_bands[:,3:6]), q_mode='q1', op_mode='up')
    y10 = sampling.q_sampling(pad(sub_bands[:,6:9]), q_mode='q1', op_mode='up')
    y11 = sampling.q_sampling(pad(sub_bands[:,9:12]), q_mode='q1', op_mode='up')
    
    y00 = padding_per1(y00)
    y10 = padding_per1(y10)
    y01 = padding_per3(y01)
    y11 = padding_per3(y11)
    
    y00 = conv2d(y00, h0, padding=0, groups=3)[:,:,0:h*2,0:w*2]
    y01 = conv2d(y01, h1, padding=0, groups=3)[:,:,0:h*2,0:w*2]
    y10 = conv2d(y10, h0, padding=0, groups=3)[:,:,0:h*2,0:w*2]
    y11 = conv2d(y11, h1, padding=0, groups=3)[:,:,0:h*2,0:w*2]
    print('y00 y01 shape', y00.shape, y01.shape)
    print('y10 y11 shape', y10.shape, y11.shape)
    
    y0 = y00 + y01
    y1 = y10 + y11
    
    y0 = sampling.q_sampling(y0, q_mode='q0', op_mode='up')
    y1 = sampling.q_sampling(y1, q_mode='q0', op_mode='up')
    
    y0 = padding_per1(y0)
    y1 = padding_per3(y1)    
    
    y0 = conv2d(y0, h0, padding=0, groups=3)[:,:,0:h*2,0:w*2]
    y1 = conv2d(y1, h1, padding=0, groups=3)[:,:,0:h*2,0:w*2]
    # print('y0 y1 shape', y0.shape, y1.shape)
    return y0+y1