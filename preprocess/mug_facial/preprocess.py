import os
import cv2
import numpy as np
import glob
import tqdm
import pdb

action_clss = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def find_face_region(frames):
    haar_face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    min_x, min_y, max_x, max_y = 5000, 5000, 0, 0
    for frame in [frames[0]]:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            tl_x = x
            tl_y = y
            br_x = x + w
            br_y = y + h
            min_x = min(min_x, tl_x)
            min_y = min(min_y, tl_y)
            max_x = max(max_x, br_x)
            max_y = max(max_y, br_y)
    return min_x, min_y, max_x, max_y


def get_videos(folder):
    videos = []
    for seq in os.listdir(folder):
        if not 'end' in seq:
            seq_abs = os.path.join(folder, seq)
            if os.path.isdir(seq_abs):
                frames = []
                for file in sorted(glob.glob(os.path.join(seq_abs, '*.jpg'))):
                    frame = cv2.imread(file)
                    frames.append(frame)
                if len(frames) < 40:
                    continue
                tl_w, tl_h, br_w, br_h = find_face_region(frames)
                video = np.stack(frames, axis=0)[:, :, :, ::-1]  # BGR -> RGB
                video = video[:, tl_h: br_h, tl_w: br_w]
                videos.append(video)
    return videos


def split_video(video):
    every_nth = 4
    mini_videos = []
    for i in range(every_nth):
        samples = range(i, len(video), every_nth)
        mini_videos.append(video[samples])
    return mini_videos


def save_npy(clips, actor, action, seq_id, output_dir):
    for idx, clip in enumerate(clips):
        output_path = os.path.join(output_dir, '%s_%s_%03d_%02d.npy' % (actor, action, seq_id, idx))
        np.save(output_path, clip)


def make_gif(clips, actor, action, seq_id, output_dir):
    for idx, clip in enumerate(clips):
        for frame_id in range(len(clip)):
            frame_name = os.path.join(output_dir, '%s_%s_%03d_%02d_%03d.png' % (actor, action, seq_id, idx, frame_id))
            cv2.imwrite(frame_name, clip[frame_id][:, :, ::-1])

        os.system('rm %s' % os.path.join(output_dir, '%s_%s_%03d_%02d.gif' % (actor, action, seq_id, idx)))
        os.system('ffmpeg -f image2 -framerate 5 -i %s %s' % (os.path.join(output_dir, '%s_%s_%03d_%02d' % (actor, action, seq_id, idx) + '_%03d.png'),
                                                              os.path.join(output_dir, '%s_%s_%03d_%02d.gif' % (actor, action, seq_id, idx))))
        os.system('rm %s' % os.path.join(output_dir, '%s_%s_%03d_%02d_*.png' % (actor, action, seq_id, idx)))


def save_videos(videos, actor, action):
    npy_dir = '/scratch4/datasets/MUG_facial_expression/npy_files'
    if not os.path.isdir(npy_dir):
        os.makedirs(npy_dir)

    gif_dir = '/scratch4/datasets/MUG_facial_expression/gifs'
    if not os.path.isdir(gif_dir):
        os.makedirs(gif_dir)

    for idx, video in enumerate(videos):
        mini_videos = split_video(video)
        save_npy(mini_videos, actor, action, idx, npy_dir)
        make_gif(mini_videos, actor, action, idx, gif_dir)


dataroot = '/scratch4/datasets/MUG_facial_expression/subjects3'


for idx in os.listdir(dataroot):
    idx_abs = os.path.join(dataroot, idx)
    if os.path.isdir(idx_abs):
        for dir in os.listdir(idx_abs):
            dir_abs = os.path.join(idx_abs, dir)
            if os.path.isdir(dir_abs):
                print(dir_abs)
                if dir.startswith('session'):
                    for action in os.listdir(dir_abs):
                        action_abs = os.path.join(dir_abs, action)
                        if os.path.isdir(action_abs) and action in action_clss:
                            videos = get_videos(action_abs)
                            save_videos(videos, idx, action)
                else:
                    action = dir
                    action_abs = dir_abs
                    if action in action_clss:
                        videos = get_videos(action_abs)
                        save_videos(videos, idx, action)





