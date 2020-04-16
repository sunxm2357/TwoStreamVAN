import os
import glob
import imageio
import cv2
import numpy as np
import pdb
import tqdm


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


root = '/Users/sunxm/Downloads/mixamo/white_bg'
output_dir = '/Users/sunxm/Downloads/mixamo/white_bg_crop'
makedir(output_dir)
files = glob.glob(os.path.join(root, '*.mp4'))
mask_dir = os.path.join(output_dir, 'mask_img')
makedir(mask_dir)
frame_dir = os.path.join(output_dir, 'frame_img')
makedir(frame_dir)
with open('preprocess/Mixamo/crop_area.txt', 'w+') as f:
    f.writelines('action seq idx tl_x tl_y br_x br_y \n')

for file in tqdm.tqdm(files):
    vid = imageio.get_reader(file)
    count = 0
    total_mask = np.zeros((480, 640))
    frames = []
    # read in frame and sum up mask
    while True:
        try:
            frame = vid.get_data(count)
        except:
            break
        count += 1
        frames.append(frame)
        avg_frame = np.mean(frame, axis=-1)
        mask = (avg_frame < 250).astype(int)
        mask_path = os.path.join(mask_dir, '%04d.png' % count)
        cv2.imwrite(mask_path, mask * 255)
        frame_path = os.path.join(frame_dir, '%04d.png' % count)
        cv2.imwrite(frame_path, frame[:, :, ::-1])
        total_mask = mask + total_mask

    # find the tl and br for the crop
    motion_area = np.where(total_mask != 0)
    tl_x, tl_y, br_x, br_y = motion_area[0].min(), motion_area[1].min(), motion_area[0].max(), motion_area[1].max()
    center = (tl_x + br_x)/2, (tl_y + br_y)/2
    h, w = br_x - tl_x, br_y - tl_y
    if w > 480:
        max_offset_y = max(br_y - 480, tl_y)
        max_appearance = 0
        for y in range(tl_y, max_offset_y):
            appearance = np.sum(total_mask[:, y: y + 480])
            if appearance > max_appearance:
                max_appearance = appearance
                tl_y = y
        tl_x = 0
        br_x = 480
        br_y = min(tl_y + 480, 640)
    else:
        l = max(h, w)
        center = min(max(center[0], l/2), 480 - l/2), min(max(center[1], l / 2), 640 - l / 2)
        tl_x, tl_y = center[0] - l/2, center[1] - l/2
        br_x, br_y = tl_x + l, tl_y + l

    # get the numpy file for the video
    video = np.array(frames)
    cropped_video = video[:, tl_x: br_x, tl_y:br_y]

    # get video name
    video_name = file.split('/')[-1].split('.')[0]

    # # save to gif
    # for idx, frame in enumerate(cropped_video):
    #     frame_path = os.path.join(output_dir, '%s_%03d.png' % (video_name, idx))
    #     cv2.imwrite(frame_path, frame[:, :, ::-1])
    # cmd1 = 'rm %s' % os.path.join(output_dir, '%s.gif' % video_name)
    # # cmd2 = 'ffmpeg -f image2 -framerate 30 -filter_complex \"paletteuse\" -i %s %s' % (os.path.join(output_dir, video_name + '_%03d.png'),
    # #                                                    os.path.join(output_dir, '%s.gif' % video_name))
    # cmd2 = "gifski --fps 30 -o %s %s" % (os.path.join(output_dir, '%s.gif' % video_name),
    #                                      os.path.join(output_dir, video_name + '_*.png'))
    # cmd3 = 'rm %s' % os.path.join(output_dir, video_name + '*.png')
    # os.system(cmd1)
    # os.system(cmd2)
    # os.system(cmd3)

    # get action class, seq, idx

    _, action, seq, idx = video_name.split('_')
    with open('preprocess/Mixamo/crop_area.txt', 'a') as f:
        for i in range(15):
            if i % 3 == int(idx):
                f.writelines('%s %s %d %d %d %d %d \n' % (action, seq, i, tl_x, tl_y, br_x, br_y))








