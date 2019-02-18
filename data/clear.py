import os
import shutil


def split_data(label, data_dir, des_dir):
    with open(label) as f:
        for line in f:
            line = line.strip()

            if not os.path.exists(des_dir + os.path.dirname(line)):
                os.makedirs(des_dir + os.path.dirname(line))

            shutil.copyfile(data_dir + line, des_dir + line)
    print('Done')


def clear_data():
    data_dir = '/Users/panda/Downloads/ready_chinese_food'
    dir = '/Users/panda/Code/Source/Hierachy/data'

    labels = [dir+'/'+x for x in ['TR.txt', 'TE.txt', 'VAL.txt']]
    des_dirs = [ os.path.join(dir, x) for x in ['train', 'val', 'test']]

    for label, des_dir in zip(labels, des_dirs):
        split_data(label, data_dir, des_dir)

