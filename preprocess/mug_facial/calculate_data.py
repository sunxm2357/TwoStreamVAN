import os
import numpy as np

action_clss = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def update_action_info(folder, action_info):
    for seq in os.listdir(folder):
        seq_abs = os.path.join(folder, seq)
        if os.path.isdir(seq_abs):
            seq_info = {'name': seq}
            seq_len = len([name for name in os.listdir(seq_abs) if name.endswith('jpg')])
            seq_info['video_len'] = seq_len
            action_info[seq] = seq_info
    return action_info


def count_seq_num(dataset):
    total_clss_c = 0
    unqualified_files = 0
    for k, v in dataset.items():
        for kk, vv in v.items():
            nums = len(vv.keys())
            total_clss_c += 1
            if nums < 2:
                print('%s %s num_seq=%d' % (k, kk, nums))
                unqualified_files += 1
    print('%d / %d = %.1f' % (unqualified_files, total_clss_c, float(unqualified_files)/total_clss_c * 100))


def get_seq_len(dataset):
    lens = []
    for k, v in dataset.items():
        for kk, vv in v.items():
            tmp_lens = []
            for kkk, vvv in vv.items():
                tmp_lens.append(vvv['video_len'])
            nums = len(tmp_lens)
            top2 = np.sort(tmp_lens)[max(0, nums-1):].tolist()
            lens += top2
            if np.min(top2) < 50:
                print(k, kk, vv)
    unqualified_lens = [l for l in lens if l < 50]
    print('%d / %d = %.1f' % (len(unqualified_lens), len(lens), float(len(unqualified_lens))/len(lens) * 100))


dataroot = '/scratch4/datasets/MUG_facial_expression/subjects3'

dataset_info = {}
for idx in os.listdir(dataroot):
    idx_abs = os.path.join(dataroot, idx)
    if os.path.isdir(idx_abs):
        actor_info = dataset_info.get(idx, {})
        for dir in os.listdir(idx_abs):
            dir_abs = os.path.join(idx_abs, dir)
            if os.path.isdir(dir_abs):
                if dir.startswith('session'):
                    for action in os.listdir(dir_abs):
                        action_abs = os.path.join(dir_abs, action)
                        if os.path.isdir(action_abs) and action in action_clss:
                            action_info = actor_info.get(action, {})
                            action_info = update_action_info(action_abs, action_info)
                            actor_info[action] = action_info
                else:
                    action = dir
                    action_abs = dir_abs
                    if action in action_clss:
                        action_info = actor_info.get(action, {})
                        action_info = update_action_info(action_abs, action_info)
                        actor_info[action] = action_info
        dataset_info[idx] = actor_info


count_seq_num(dataset_info)
get_seq_len(dataset_info)




