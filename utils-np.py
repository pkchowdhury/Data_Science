# %% [code]

import numpy as np
from PIL import Image

def dummy_loader(model_path):
    '''
    Load a stored keras model and return its weights.
    
    Input
    ----------
        The file path of the stored keras model.
    
    Output
    ----------
        Weights of the model.
        
    '''
    backbone = keras.models.load_model(model_path, compile=False)
    W = backbone.get_weights()
    return W

def image_to_array(arrays):
    '''
    Converting RGB images represented as ndarrays to a single numpy array.
    
    Input
    ----------
        arrays: an iterable of ndarrays representing images
        
    Output
    ----------
        An array with shape = (num_arrays, height, width, channels)
        
    '''
    
    # number of arrays
    num_arrays = len(arrays)
    
    # assume all arrays have the same shape
    size, _, channel = arrays[0].shape
    
    # allocation
    out = np.empty((num_arrays, size, size, channel))
    
    for i, arr in enumerate(arrays):
        out[i] = arr
        
    return out

def shuffle_ind(L):
    '''
    Generating random shuffled indices.
    
    Input
    ----------
        L: an int that defines the largest index
        
    Output
    ----------
        a numpy array of shuffled indices with shape = (L,)
    '''
    
    ind = np.arange(L)
    np.random.shuffle(ind)
    return ind

def freeze_model(model, freeze_batch_norm=False):
    '''
    freeze a keras model
    
    Input
    ----------
        model: a keras model
        freeze_batch_norm: False for not freezing batch notmalization layers
    '''
    if freeze_batch_norm:
        for layer in model.layers:
            layer.trainable = False
    else:
        from tensorflow.keras.layers import BatchNormalization    
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
    return model


