import os
from shutil import copy, rmtree
import random


def make_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


random.seed(0)
# 20% of the data set divided into test sets
split_test_rate = 0.2
# data folder
data_path = r'.\data-process\bengin-and-maligant\400X'
# Test Data Folder
data_root = r'.\train_test_data\400X'


data_class = [cla for cla in os.listdir(data_path)]
print("数据的种类分别为：")
print(data_class)
# 建立保存训练集的文件夹
train_data_root = os.path.join(data_root, "train")
make_file(train_data_root)
for num_class in data_class:
    # Create folders corresponding to each category
    make_file(os.path.join(train_data_root, num_class))

test_data_root = os.path.join(data_root, "test")
make_file(test_data_root)
for num_class in data_class:
    make_file(os.path.join(test_data_root, num_class))
for num_class in data_class:
    num_class_path = os.path.join(data_path, num_class)
    images = os.listdir(num_class_path)
    num = len(images)
    test_index = random.sample(images, k=int(num * split_test_rate))
    for index, image in enumerate(images):
        if image in test_index:
            data_image_path = os.path.join(num_class_path, image)
            val_new_path = os.path.join(test_data_root, num_class)
            copy(data_image_path, val_new_path)
        else:
            data_image_path = os.path.join(num_class_path, image)
            train_new_path = os.path.join(train_data_root, num_class)
            copy(data_image_path, train_new_path)
    print("\r[{}] split_rating [{}/{}]".format(num_class, index + 1, num), end="")  # processing bar
    print()

print("       ")
print("       ")
print("划分成功")
