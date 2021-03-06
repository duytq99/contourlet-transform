import torch
import numpy as np
from torch.nn.functional import affine_grid, grid_sample


def q_sampling(img, q_mode='q0', op_mode='down'):
    h,w = img.shape[2], img.shape[3]
    pad = torch.nn.ReflectionPad2d((w//2,w//2,h//2,h//2))
    img = pad(img)
    
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
    
    h,w = img.shape[2], img.shape[3]
    img = img[:,:,h//4:3*h//4,w//4:3*w//4]
    return img

if __name__ == '__main__':
    
    
    def to_image(tensor):
        return tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    import cv2 
    tensor = torch.tensor(cv2.imread('image/lena.png',1) /255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    cv2.imshow('original',to_image(tensor))
    print(tensor.shape)
    
    _, _, h, w = tensor.shape
    # padding_per = torch.nn.ReflectionPad2d((w//2,w//2,h//2,h//2))
    padding_per = torch.nn.ZeroPad2d((w//2,w//2,h//2,h//2))
    
    tensor = padding_per(tensor)
    print(tensor.shape)
    tensor_down = q_sampling(tensor, q_mode='q0', op_mode='up')
    cv2.imshow('a' ,to_image(tensor_down))
    tensor_down = q_sampling(tensor_down, q_mode='q1', op_mode='up')

    print(tensor_down.shape)
    cv2.imshow('up/down' ,to_image(tensor_down))
    
    from skimage import metrics
    # print(metrics.peak_signal_noise_ratio(tensor.numpy(), tensor_down.numpy()))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    