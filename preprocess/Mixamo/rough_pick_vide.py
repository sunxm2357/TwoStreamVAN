import sys
sys.path.insert(0, '..')
import os
import glob
from utils.util import makedir
import pdb

phrase_list = ['clapping', 'waving', 'pick fruit', 'kick soccerball', 'wheelbarrow', 'stall soccerball', 'baseball step',
               'rifle side', 'kneeing', 'jazz dancing', 'walking', 'running', 'jump', 'kick', 'hook', 'cheering while',
               'goalkeeper catch', 'throw', 'climb']

root = '/Users/sunxm/Downloads/mixamo/'
files = glob.glob(os.path.join(root, '*'))
output_path = '/Users/sunxm/Documents/mixamo/rough_pick/'
makedir(output_path)
for file_path in files:
    file_name = file_path.split('/')[-1]
    for phrase in phrase_list:
        words = phrase.split()
        flag = True
        for word in words:
            if word not in file_name.lower():
                flag = False
                break
        if flag:
            # pdb.set_trace()
            # tokens = file_path.split()

            file_path = file_path.replace(' ', '\ ')
            file_path = file_path.replace('(', '\(')
            file_path = file_path.replace(')', '\)')
            # new_file_path = ''
            # for idx, token in enumerate(tokens):
            #     if idx != len(tokens)-1:
            #         new_file_path += token + '\ '
            #     else:
            #         new_file_path += token
            # pdb.set_trace()
            print(file_path)
            os.system('cp %s %s' % (file_path, output_path))
            break


