import os
import argparse

actions = ['bow', 'cheer', 'clap', 'climb',
           'crossstep', 'hit', 'hook', 'jump',
           'kick', 'knee', 'pick', 'pitch',
           'push', 'roll', 'run', 'squat',
           'stall', 'standup', 'walk', 'wave']

camera = ['middle', 'left', 'right']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='the path of the generated videos models')
    parser.add_argument('--actor', type=str, help='the actor name')
    args = parser.parse_args()

    for action in actions:
        for video_idx in range(1,3):
            for idx in range(15):
                file_name = '%s_%s_%03d_%d.mp4' % (args.actor, action, video_idx, idx)
                filepath = os.path.join(args.datapath, file_name)
                if not os.path.exists(filepath):
                    print('Actor: %s, Action: %s, Video: %03d, Bg %d, Camera %s is missing'%
                          (args.actor, action, video_idx, idx//3, camera[int(idx % 3)]))


if __name__ == "__main__":
    main()

