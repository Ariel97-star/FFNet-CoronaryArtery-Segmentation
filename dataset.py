import os
import argparse
import random
from shutil import copytree
import shutil

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)

def dataset_alloc():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)

    # data path
    parser.add_argument('--origin_data_path', type=str, default="./GMCTA/image/")
    parser.add_argument('--origin_GT_path', type=str, default="./GMCTA/label/")

    parser.add_argument('--train_path', type=str, default="./dataset/train/")
    parser.add_argument('--train_GT_path', type=str, default="./dataset/train_GT/")
    parser.add_argument('--valid_path', type=str, default="./dataset/valid/")
    parser.add_argument('--valid_GT_path', type=str, default="./dataset/valid_GT/")
    parser.add_argument('--test_path', type=str, default="./dataset/test/")
    parser.add_argument('--test_GT_path', type=str, default="./dataset/test_GT/")

    config = parser.parse_args()

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    names = os.listdir(config.origin_data_path)
    namenum_total = len(names)

    namenum_train = int(config.train_ratio * namenum_total)
    namenum_valid = int(config.valid_ratio * namenum_total)+1
    namenum_test = namenum_total - namenum_train - namenum_valid

    Arange = list(range(namenum_total))
    random.shuffle(Arange)

    for i in range(namenum_train):
        idn = Arange.pop()
        src = os.path.join(config.origin_data_path, names[idn])
        dst = os.path.join(config.train_path, names[idn])
        copytree(src, dst)

        src = os.path.join(config.origin_GT_path, names[idn])
        dst = os.path.join(config.train_GT_path, names[idn])
        copytree(src, dst)


    for i in range(namenum_valid):
        idn = Arange.pop()
        src = os.path.join(config.origin_data_path, names[idn])
        dst = os.path.join(config.valid_path, names[idn])
        copytree(src, dst)

        src = os.path.join(config.origin_GT_path, names[idn])
        dst = os.path.join(config.valid_GT_path, names[idn])
        copytree(src, dst)


    for i in range(namenum_test):
       idn = Arange.pop()
       src = os.path.join(config.origin_data_path, names[idn])
       dst = os.path.join(config.test_path, names[idn])
       copytree(src, dst)

       src = os.path.join(config.origin_GT_path, names[idn])
       dst = os.path.join(config.test_GT_path, names[idn])
       copytree(src, dst)

if __name__ == '__main__':
    dataset_alloc()
