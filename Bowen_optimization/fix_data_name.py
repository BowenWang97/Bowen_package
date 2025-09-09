import os
import scipy.io as sio
import re

path = os.path.dirname(__file__)

data = sio.loadmat(path+'\\model_params.mat')

fixed_data = {
    re.sub(r'\.', '_', key): value 
    for key, value in data.items() 
    if not key.startswith('__')
}

sio.savemat(path+'\\model_params.mat', fixed_data)