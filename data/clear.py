from os import path, makedirs
import shutil

def split_data(label, ori_dir, des_dir):
    with open(label) as f:
        for line in f:
            file = line.strip()[1:] # Starting from 1 for removing '/'

            if not path.exists(path.join(des_dir, path.dirname(file))):
                makedirs(path.join(des_dir, path.dirname(file)))

            shutil.copyfile(path.join(ori_dir, file), path.join(des_dir, file))
    print('Done')


def clear_data(data_dir):
    ori_data = 'ready_chinese_food'

    labels = ['TR.txt', 'TE.txt', 'VAL.txt']
    des_dirs = ['train', 'val', 'test']

    for label, des_dir in zip(labels, des_dirs):
        split_data(label, ori_data, des_dir)

if __name__ == '__main__':
    clear_data('./')
