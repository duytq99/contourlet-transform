import torch
import numpy as np
from torch.nn.functional import affine_grid, grid_sample


def q_sampling(img, q_mode='q0', op_mode='down'):
    if q_mode=='q0' and op_mode=='down':
        q = torch.tensor([[1, -1, 0],[1, 1, 0]])
    elif q_mode=='q1' and op_mode=='down':
        q = torch.tensor([[1, 1, 0],[-1, 1, 0]])
    elif q_mode=='q0' and op_mode=='up':
        q  = torch.tensor([[0.5, 0.5, 0],[-0.5, 0.5, 0]])
    elif q_mode=='q1' and op_mode=='up':
        q = torch.tensor([[0.5, -0.5, 0],[0.5, 0.5, 0]])
    else:
        raise NotImplementedError("Not available q type")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    q = q[None, ...].type(torch.FloatTensor).repeat(img.shape[0],1,1)
    grid = affine_grid(q, img.size(), align_corners=True).type(torch.FloatTensor).to(device)
    img = grid_sample(img, grid, align_corners=True)
    return img

if __name__ == '__main__':
    tensor = torch.rand(1,1,512,512)
    tensor = q_sampling(tensor, q_mode='q0', op_mode='up')
    print(tensor.shape)