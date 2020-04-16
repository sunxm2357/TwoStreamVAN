import os
import argparse
import pdb


def makedir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

action_class = {'air squat bent arms': 'squat',
                'air squat': 'squat',
                'baseball hit': 'hit',
                'baseball strike': 'hit',
                'baseball pitching (1)': 'pitch',
                'baseball pitching': 'pitch',
                'cheering while sitting (1)': 'cheer',
                'cheering while sitting': 'cheer',
                'clapping (1)': 'clap',
                'clapping': 'clap',
                'climbing (1)': 'climb',
                'climbing': 'climb',
                'crouch walk left': 'crossstep',
                'standing walk left': 'crossstep',
                'hook': 'hook',
                'hook punch': 'hook',
                'jumping': 'jump',
                'standing jump': 'jump',
                'kicking': 'kick',
                'roundhouse kick': 'kick',
                'kneeing soccerball (1)': 'knee',
                'kneeing soccerball': 'knee',
                'pick fruit (1)': 'pick',
                'pick fruit': 'pick',
                'prone roll': 'roll',
                'roll left': 'roll',
                'running': 'run',
                'treadmill running': 'run',
                'stall soccerball': 'stall',
                'stall soccerball (1)': 'stall',
                'start walking (1)': 'walk',
                'start walking': 'walk',
                'waving (1)': 'wave',
                'waving': 'wave',
                'zombie stand up (1)': 'standup',
                'zombie stand up': 'standup',
                'quick formal bow': 'bow',
                'quick informal bow': 'bow',
                'push start': 'push',
                'push stop': 'push',
                }

appearance = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='the path of downloaded models')
    parser.add_argument('--output_path', type=str, help='the path of grouped models, one level above the actor name')
    args = parser.parse_args()

    for file in os.listdir(args.input_path):
        if file.endswith('.fbx'):
            # separate the actor and action name, e.g. swat@Air Squat Bent Arms.fbx
            tokens = file.split('.')[0].split('@')
            # get the actor name
            actor = tokens[0].lower()
            # get the action name
            action = tokens[1].lower()
            action_clss = action_class.get(action, None)
            file = file.replace(' ', '\ ').replace('(', '\(').replace(')', '\)')
            file_path = os.path.join(args.input_path, file)
            if action_clss is None:

                os.system('rm %s' % file_path)
            else:
                actor_appearance = appearance.get(actor, {})
                actor_appearance[action] = actor_appearance.get(action, 0) + 1
                appearance[actor] = actor_appearance
                output_dir = os.path.join(args.output_path, actor, action_clss)
                makedir(output_dir)
                os.system('cp %s %s' % (file_path, output_dir))

    for actor in appearance.keys():
        for action in action_class.keys():
            if action not in appearance[actor].keys():
                print('For actor \'%s\', actor \' %s \' is missing' % (actor, action))


if __name__ == '__main__':
    main()