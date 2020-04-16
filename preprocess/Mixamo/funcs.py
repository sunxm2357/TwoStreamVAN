import sys
sys.path.insert(0, '..')
import numpy as np
from copy import deepcopy
import glob
import os
import imageio
import cv2
import tqdm
from numpy import linalg as LA
import pdb


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def remove_watermark(img, location=None, mask=None, nn=None, mode='Down'):
    """
    :param img: [h, w, c]
    :param location: a tuple of (tl_x, tl_y, br_x, br_y)
    :param mode: Down or Around
    :return:
    """
    if mode == "Down":
        tl_x, tl_y, br_x, br_y = location
        h = br_x - tl_x
        w = br_y - tl_y
        stripe = img[-1, tl_y: br_y]
        new_patch = np.repeat(np.expand_dims(stripe, axis=0), h, axis=0)
    elif mode == 'Around':
        tl_x, tl_y, br_x, br_y = location
        h = br_x - tl_x
        w = br_y - tl_y
        boundary = deepcopy(img[tl_x - 1:br_x + 1, tl_y - 1:br_y + 1])
        boundary[1:-1, 1:-1] = 0
        avg_color = np.sum(np.sum(boundary, axis=0), axis=0) / (2 * h + 2 * w + 4)
        avg_color = np.expand_dims(np.expand_dims(avg_color, axis=0), axis=0)
        new_patch = np.repeat(np.repeat(avg_color, h, axis=0), w, axis=1)
    elif mode == 'NN':
        new_img = deepcopy(img)
        area = np.where(mask)
        for idx in range(len(area[0])):
            x, y = area[0][idx], area[1][idx]
            new_img[x, y] = img[int(nn[x, y][0]), int(nn[x, y][1])]
        img = new_img
    else:
        raise ValueError('mode %s for removing watermark is not defined' % mode)

    # img[tl_x:br_x, tl_y:br_y] = new_patch
    return img


def image_process(img, seq_idx, is_NN=False, mask=None, nn=None):
    if is_NN:
        img = remove_watermark(img, mask=mask, nn=nn, mode='NN')
    else:
        location = (424, 18, 464, 53)
        if seq_idx in [8, 12]:
            img = remove_watermark(img, location=location, mode='Around')
        else:
            img = remove_watermark(img, location=location, mode='Down')
    return img


def NN(mask):
    h, w = mask.shape
    area = np.where(mask)
    tl_x, tl_y, br_x, br_y = area[0].min(), area[1].min(), area[0].max(), area[1].max()
    x_axis = np.repeat(np.expand_dims(np.arange(0, h), axis=-1), w, axis=-1)
    y_axis = np.repeat(np.expand_dims(np.arange(0, w), axis=0), h, axis=0)
    coords = np.stack((x_axis, y_axis), axis=-1)
    min_distance = mask.astype('int') * 100000 * np.ones((h, w))
    nn_coord = np.zeros((h, w, 2))
    print(tl_x-1, br_x + 2, tl_y-1, br_y +2)
    for x in tqdm.tqdm(range(tl_x-1, br_x + 2)):
        for y in range(tl_y-1, br_y + 2):
            if not mask[x, y]:
                tile_loc = np.repeat(np.repeat(coords[x, y].reshape(1, 1, -1), h, axis=0), w, axis=1)
                grid_dis = coords - tile_loc
                l2_distance = LA.norm(grid_dis, axis=-1)
                l2_distance = mask.astype('int') * l2_distance
                update_mask = (l2_distance < min_distance).astype('int')
                coord_mask = np.repeat(np.expand_dims(update_mask, axis=-1), 2, axis=-1)
                min_distance = update_mask * l2_distance + (1 - update_mask) * min_distance
                nn_coord = coord_mask * tile_loc + (1 - coord_mask) * nn_coord
    return nn_coord


if __name__ == '__main__':

    # test image
    test_video = '/Users/sunxm/Downloads/mixamo/white_bg/test_2.mp4'
    vid = imageio.get_reader(test_video)
    test_image = np.array(vid.get_data(0))
    watermark_mask = (test_image.sum(axis=-1) > 0)
    vid.close()
    nn = NN(watermark_mask)

    root = '/Users/sunxm/Downloads/mixamo/videos/douglas/'
    output_dir = '/Users/sunxm/Downloads/mixamo/videos/test/'
    makedir(output_dir)
    files = glob.glob(os.path.join(root, '*.mp4'))
    for video in tqdm.tqdm(files):
        vid = imageio.get_reader(video)
        frame0 = vid.get_data(0)
        video_name = video.split('/')[-1].split('.')[0]
        seq_idx = int(video_name.split('_')[-1])
        frame0 = image_process(frame0, seq_idx, is_NN=True, mask=watermark_mask, nn=nn)
        filename = os.path.join(output_dir, video_name+'.png')
        cv2.imwrite(filename, frame0[:, :, ::-1])
