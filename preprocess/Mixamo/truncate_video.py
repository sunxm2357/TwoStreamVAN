import os
import glob
import numpy as np

root = '/scratch/dataset/Syn_Action_npy/'
files = glob.glob(os.path.join(root, '*.npy'))

for path in files:
    if 'standup_001' in path:
        npy = np.load(path)
        np.save(path, npy[:86])
    elif 'wave_001' in path:
        npy = np.load(path)
        np.save(path, npy[:75])