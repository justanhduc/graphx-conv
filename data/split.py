import argparse
import numpy as np
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='path to the raw ShapeNet db')
args = parser.parse_args()

train_file = 'train_obj.txt'
test_file = 'test_obj.txt'
db_path = args.path


def _make_save_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    with open(train_file, 'r') as f:
        train_list = f.readlines()

    with open(test_file, 'r') as f:
        test_list = f.readlines()

    print('Making training folder...')
    for train_data in train_list:
        tr_data = train_data.replace('\n', '').split('/')[-1]
        category = tr_data.split('_')[0]
        tr_save_path = os.path.join(db_path, category, 'train')
        _make_save_directory(tr_save_path)

        if os.path.exists(class_path + tr_data):
            destination = tr_save_path + tr_data
            print(destination)
            shutil.move(os.path.join(db_path, tr_data), destination)

    print('Making testing folder...')
    for test_data in test_list:
        ts_data = test_data.replace('\n', '').split('/')[-1]
        category = ts_data.split('_')[0]
        ts_save_path = os.path.join(db_path, category, 'test')
        _make_save_directory(ts_save_path)

        if os.path.exists(class_path + ts_data):
            destination = ts_save_path + ts_data
            print(destination)
            shutil.move(os.path.join(db_path, ts_data), destination)
