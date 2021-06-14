import numpy as np
import torch


def lp_filters():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    h = np.array([.037828455506995, -.023849465019380, -.11062440441842, .37740285561265])
    h = np.hstack((h, .85269867900940, h[::-1]))
    h = np.expand_dims(h, 1).astype(np.float32)
    h = torch.from_numpy(h*h.T).unsqueeze(0)

    g = np.array([-.064538882628938, -.040689417609558, .41809227322221])
    g = np.hstack((g, .78848561640566, g[::-1]))
    g = np.expand_dims(g, 1).astype(np.float32)
    g = torch.from_numpy(g*g.T).unsqueeze(0)
    return h.expand(3, 1, 9, 9).to(device), g.expand(3, 1, 7, 7).to(device)


def dfb_filters(mode = None, name=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if name=='haar':
        if mode=='r':
            g0 = np.array([[1, 1]]).astype(np.float32)/np.sqrt(2)
            g0 = torch.from_numpy(g0).expand(3, 1, 1, 2)
            g1 = np.array([[1, -1]]).astype(np.float32)/np.sqrt(2)
            g1 = torch.from_numpy(g1).expand(3, 1, 1, 2)
            return g0.to(device), g1.to(device)
        elif mode=='d':
            h0 = np.array([[1, 1]]).astype(np.float32)/np.sqrt(2)
            h0 = torch.from_numpy(h0).expand(3, 1, 1, 2)
            h1 = np.array([[-1, 1]]).astype(np.float32)/np.sqrt(2)
            h1 = torch.from_numpy(h1).expand(3, 1, 1, 2) 
            return h0.to(device), h1.to(device)
        else:
            raise NotImplementedError("Mode is not available")
    elif name=='thanh':
        if mode=='r':
            g0 = np.array([[0, -1, 0],
                        [-1, -4, -1],
                        [0, -1, 0]]).astype(np.float32)/4.0
            g0 = torch.from_numpy(g0).expand(3, 1, 3, 3)
            g1 = np.array([[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, -2, -4, -2, 0],
                        [0, 0, -1, -4, 28, -4, -1],
                        [0, 0, 0, -2, -4, -2, 0],
                        [0, 0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0]]).astype(np.float32)/32.0
            g1 = torch.from_numpy(g1).expand(3, 1, 7, 7)
            return g0.to(device), g1.to(device)
        elif mode=='d':
            h0 = np.array([[0, 0, -1, 0, 0],
                    [0, -2, 4, -2, 0],
                    [-1, 4, 28, 4, -1],
                    [0, -2, 4, -2, 0],
                    [0, 0, -1, 0, 0]]).astype(np.float32)/32.0
            h0 = torch.from_numpy(h0).expand(3, 1, 5, 5)
        
            h1 = np.array([[0, 0, 0, 0, 0 ],
                    [0, -1, 0, 0, 0],
                    [-1, 4, -1, 0, 0],
                    [0, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0]]).astype(np.float32)/4.0
            h1 = torch.from_numpy(h1).expand(3, 1, 5, 5)
            return h0.to(device), h1.to(device)
        else:
            raise NotImplementedError("Mode is not available")
    else:
        raise NotImplementedError("Filters haven't implemented")

if __name__ == '__main__':
    h, g = lp_filters()
    print('9-7 laplacian pyramid filters: ')
    print('h shape: ', h.shape)
    print(h)
    print('g shape: ', g.shape)
    print(g)
    
    mode = 'r'
    name = 'haar'
    h0, h1 = dfb_filters(mode = mode, name=name)
    print('DFB filters')
    print('mode decompose' if mode=='d' else 'mode recompose')
    print('haar filters' if name=='haar' else 'thanh filters')
    print('h0 shape: ', h0.shape)
    print(h0)
    print('h1 shape: ', h1.shape)
    print(h1)