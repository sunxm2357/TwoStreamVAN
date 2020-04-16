import os
import numpy as np
import cv2
import glob
import tqdm


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


root = '/Volumes/backup/mixamo_dataset/cropped_npy/stefani'
files = glob.glob(os.path.join(root, '*.npy'))

output_dir = '/Volumes/backup/mixamo_dataset/resized_npy'
makedir(output_dir)

for name in tqdm.tqdm(files):
    video = np.load(name)
    resized_frames = []
    for frame in video:
        resize_frame = cv2.resize(frame, (71, 71))
        resized_frames.append(resize_frame)
    resized_video = np.stack(resized_frames, axis=0)

    video_name = name.split('/')[-1]
    np.save(os.path.join(output_dir, video_name), resized_video)
