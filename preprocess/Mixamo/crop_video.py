import numpy as np
import tqdm
from funcs import *


def read_seq(loc_file):
    locs = {}
    with open(loc_file, 'r+') as f:
        lines = f.readlines()

    # pick 001 1 24 169 464 609
    for line in lines[1:]:
        action, seq, idx, tl_x, tl_y, br_x, br_y = line.split()
        action_value = locs.get(action, {})
        seq_value = action_value.get(seq, {})
        seq_value[idx] = (tl_x, tl_y, br_x, br_y)
        action_value[seq] = seq_value
        locs[action] = action_value

    return locs


def get_water_loc():
    # test image
    test_video = '/Users/sunxm/Downloads/mixamo/test_2.mp4'
    vid = imageio.get_reader(test_video)
    test_image = np.array(vid.get_data(0))
    watermark_mask = (test_image.sum(axis=-1) > 0)
    vid.close()
    nn = NN(watermark_mask)
    return watermark_mask, nn


def read_all_frames(video):
    frames = []
    vid = imageio.get_reader(video)
    count = 0
    while True:
        try:
            frame = vid.get_data(count)
        except:
            break
        count += 1
        frames.append(frame)
    return frames


def seq_remove_watermark(frames, watermark_mask, nn):
    for idx, frame in tqdm.tqdm(enumerate(frames)):
        frames[idx] = remove_watermark(frame, mask=watermark_mask, nn=nn, mode="NN")
    return frames


def crop_video(video_npy, video, locs):
    video_name = video.split('/')[-1].split('.')[0]
    actor, action, seq, idx = video_name.split('_')
    crop_coord = locs[action][seq][idx]
    cropped_video_npy = video_npy[:, int(crop_coord[0]):int(crop_coord[2]), int(crop_coord[1]): int(crop_coord[3])]
    return cropped_video_npy


def generate_video(cropped_video_npy, video, output_dir):
    video_name = video.split('/')[-1].split('.')[0]
    for idx, frame in enumerate(cropped_video_npy):
        frame_path = os.path.join(output_dir, '%s_%03d.png' % (video_name, idx))
        cv2.imwrite(frame_path, frame[:, :, ::-1])
    cmd1 = 'rm %s' % os.path.join(output_dir, '%s.gif' % video_name)
    # cmd2 = 'ffmpeg -f image2 -framerate 30 -filter_complex \"paletteuse\" -i %s %s' % (os.path.join(output_dir, video_name + '_%03d.png'),
    #                                                    os.path.join(output_dir, '%s.gif' % video_name))
    cmd2 = "gifski --fps 30 -o %s %s" % (os.path.join(output_dir, '%s.gif' % video_name),
                                         os.path.join(output_dir, video_name + '_*.png'))
    cmd3 = 'rm %s' % os.path.join(output_dir, video_name + '*.png')
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)


def save_npy(cropped_video_npy, video, output_dir):
    video_name = video.split('/')[-1].split('.')[0]
    file_path = os.path.join(output_dir, '%s.npy' % video_name)
    np.save(file_path, cropped_video_npy)


def main():
    loc_file = 'preprocess/Mixamo/crop_area.txt'
    locs = read_seq(loc_file)
    watermark_mask, nn = get_water_loc()

    root = '/Users/sunxm/Downloads/mixamo/videos/swat/'
    output_dir = '/Users/sunxm/Downloads/mixamo/videos/cropped/'
    npy_dir = '/Users/sunxm/Downloads/mixamo/videos/cropped_npy/swat/'
    makedir(output_dir)
    makedir(npy_dir)
    files = glob.glob(os.path.join(root, '*.mp4'))
    for video in tqdm.tqdm(files):
        # read in all frames of one video
        # if 'cheer_001' in video:
            frames = read_all_frames(video)
            print('Read in frames finished. total frame %d' % len(frames))
            frames = seq_remove_watermark(frames, watermark_mask, nn)
            print('Remove watermark done')
            video_npy = np.stack(frames, axis=0)
            cropped_video_npy = crop_video(video_npy, video, locs)
            generate_video(cropped_video_npy, video, output_dir)
            save_npy(cropped_video_npy, video, npy_dir)

        # frame0 = vid.get_data(0)
        # video_name = video.split('/')[-1].split('.')[0]
        # seq_idx = int(video_name.split('_')[-1])
        # frame0 = image_process(frame0, seq_idx, is_NN=True, mask=watermark_mask, nn=nn)
        # filename = os.path.join(output_dir, video_name + '.png')
        # cv2.imwrite(filename, frame0[:, :, ::-1])

if __name__ == '__main__':
    main()