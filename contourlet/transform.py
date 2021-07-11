from contourlet import filters, operations

def contourlet_transform(img, name='thanh'):
    r"""
    2 level contourlet transform
    Input: image tensor shape (batch, channels, height, width)
    Output: edge-image tensor shape (batch, channels*4, height, width)
    """
    low_band, sub_bands = contourlet_decompose(img, name=name)
    img_ = contourlet_recompose(low_band, sub_bands, name=name)
  
    return img_

def contourlet_decompose(img, name='thanh'):
    # 9-7 filters
    h, g = filters.lp_filters()
    # Laplacian Pyramid decompose
    low_band, high = operations.lp_dec(img, h, g)
    # DFB filters
    h0, h1 = filters.dfb_filters(mode='d', name=name)
    # DFB decompose
    sub_bands = operations.dfb_dec(high, h0, h1, name=name)
    return low_band, sub_bands

def contourlet_recompose(low_band, sub_bands, name='thanh'): 
    # DFB filters
    g0, g1 = filters.dfb_filters(mode='r', name=name)
    # DFB recompose
    high = operations.dfb_rec(sub_bands, g0, g1, name=name)
    # 9-7 filters
    h, g = filters.lp_filters()
    # Laplacian recompose
    img = operations.lp_rec(low_band, high, h, g)
    return img 