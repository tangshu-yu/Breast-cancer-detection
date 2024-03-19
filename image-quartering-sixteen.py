import multiprocessing

import cv2
import os
import pathlib




def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + 'Created Successfully')
        return True
    else:
        print(path + 'Catalog already exists')
        return False

# Quaternion pictures
def split_img(img_file,target_path):
    img = cv2.imread(img_file)

    img_h = img.shape[0]
    img_w = img.shape[1]

    h1_half = img_h // 2
    w1_half = img_w // 2

    img_name = os.path.basename(img_file)
    for i in range(4):
        img1 = img[int(i / 2) * h1_half: h1_half * (int(i / 2) + 1), int(i % 2) * w1_half: (int(i % 2) + 1) * w1_half]

        img1_path = os.path.join(target_path, f"{img_name[:-4]}_{i}.png")
        print("spilt img:", img1_path)
        cv2.imwrite(img1_path, img1)

def split_all_img(source_path,target_path):
    for file in pathlib.Path(source_path).glob('**/*'):
        print(file)
        split_img(os.path.join(source_path, file),target_path)

if __name__ == '__main__':
    # Quartered
    source_path = r'.\train_test_data\original-drawing'

    target_path_40_bengin_train = r'.\train_test_data\quartering\40X\train\bengin'
    target_path_40_bengin_test = r'.\train_test_data\quartering\40X\test\bengin'
    mkdir(target_path_40_bengin_train)
    mkdir(target_path_40_bengin_test)

    target_path_100_bengin_train = r'.\train_test_data\quartering\100X\train\bengin'
    target_path_100_bengin_test = r'.\train_test_data\quartering\100X\test\bengin'
    mkdir(target_path_100_bengin_train)
    mkdir(target_path_100_bengin_test)

    target_path_200_bengin_train = r'.\train_test_data\quartering\200X\train\bengin'
    target_path_200_bengin_test = r'.\train_test_data\quartering\200X\test\bengin'
    mkdir(target_path_200_bengin_train)
    mkdir(target_path_200_bengin_test)

    target_path_400_bengin_train = r'.\train_test_data\quartering\400X\train\bengin'
    target_path_400_bengin_test = r'.\train_test_data\quartering\400X\test\bengin'
    mkdir(target_path_400_bengin_train)
    mkdir(target_path_400_bengin_test)

    target_path_40_maligant_train = r'.\train_test_data\quartering\40X\train\maligant'
    target_path_40_maligant_test = r'.\train_test_data\quartering\40X\test\maligant'
    mkdir(target_path_40_maligant_train)
    mkdir(target_path_40_maligant_test)

    target_path_100_maligant_train = r'.\train_test_data\quartering\100X\train\maligant'
    target_path_100_maligant_test = r'.\train_test_data\quartering\100X\test\maligant'
    mkdir(target_path_100_maligant_train)
    mkdir(target_path_100_maligant_test)

    target_path_200_maligant_train = r'.\train_test_data\quartering\200X\train\maligant'
    target_path_200_maligant_test = r'.\train_test_data\quartering\200X\test\maligant'
    mkdir(target_path_200_maligant_train)
    mkdir(target_path_200_maligant_test)

    target_path_400_maligant_train = r'.\train_test_data\quartering\400X\train\maligant'
    target_path_400_maligant_test = r'.\train_test_data\quartering\400X\test\maligant'
    mkdir(target_path_400_maligant_train)
    mkdir(target_path_400_maligant_test)

    for root, dirs, files in os.walk(source_path):
        file_name_list = root.split('\\')
        if file_name_list[-1] == 'bengin':
            if file_name_list[-3] == '40X':
                if file_name_list[-2] == "train":
                    # split_all_img(root,target_path_40_bengin_train)
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_bengin_test))
                process.start()
            if file_name_list[-3] == '100X':
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_bengin_test))
                process.start()
            if file_name_list[-3] == '200X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\200X\bengin')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_bengin_test))
                process.start()
            if file_name_list[-3] == '400X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\400X\bengin')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_bengin_test))
                process.start()
        elif file_name_list[-1] == 'maligant':
            if file_name_list[-3] == '40X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\40X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_maligant_test))
                process.start()
            if file_name_list[-3] == '100X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\100X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_maligant_test))
                process.start()
            if file_name_list[-3] == '200X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\200X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_maligant_test))
                process.start()
            if file_name_list[-3] == '400X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\400X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_maligant_test))
                process.start()

    print('Quadruple completion')
    # To four equal parts and four equal parts again to 16 equal parts #
    source_path = r'.\train_test_data\quartering'
    target_path_40_bengin_train = r'.\train_test_data\sixteen\40X\train\bengin'
    target_path_40_bengin_test = r'.\train_test_data\sixteen\40X\test\bengin'
    mkdir(target_path_40_bengin_train)
    mkdir(target_path_40_bengin_test)

    target_path_100_bengin_train = r'.\train_test_data\sixteen\100X\train\bengin'
    target_path_100_bengin_test = r'.\train_test_data\sixteen\100X\test\bengin'
    mkdir(target_path_100_bengin_train)
    mkdir(target_path_100_bengin_test)

    target_path_200_bengin_train = r'.\train_test_data\sixteen\200X\train\bengin'
    target_path_200_bengin_test = r'.\train_test_data\sixteen\200X\test\bengin'
    mkdir(target_path_200_bengin_train)
    mkdir(target_path_200_bengin_test)

    target_path_400_bengin_train = r'.\train_test_data\sixteen\400X\train\bengin'
    target_path_400_bengin_test = r'.\train_test_data\sixteen\400X\test\bengin'
    mkdir(target_path_400_bengin_train)
    mkdir(target_path_400_bengin_test)

    target_path_40_maligant_train = r'.\train_test_data\sixteen\40X\train\maligant'
    target_path_40_maligant_test = r'.\train_test_data\sixteen\40X\test\maligant'
    mkdir(target_path_40_maligant_train)
    mkdir(target_path_40_maligant_test)

    target_path_100_maligant_train = r'.\train_test_data\sixteen\100X\train\maligant'
    target_path_100_maligant_test = r'.\train_test_data\sixteen\100X\test\maligant'
    mkdir(target_path_100_maligant_train)
    mkdir(target_path_100_maligant_test)

    target_path_200_maligant_train = r'.\train_test_data\sixteen\200X\train\maligant'
    target_path_200_maligant_test = r'.\train_test_data\sixteen\200X\test\maligant'
    mkdir(target_path_200_maligant_train)
    mkdir(target_path_200_maligant_test)

    target_path_400_maligant_train = r'.\train_test_data\sixteen\400X\train\maligant'
    target_path_400_maligant_test = r'.\train_test_data\sixteen\400X\test\maligant'
    mkdir(target_path_400_maligant_train)
    mkdir(target_path_400_maligant_test)
    for root, dirs, files in os.walk(source_path):
        file_name_list = root.split('\\')
        if file_name_list[-1] == 'bengin':
            if file_name_list[-3] == '40X':
                if file_name_list[-2] == "train":
                    # split_all_img(root,target_path_40_bengin_train)
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_bengin_test))
                process.start()
            if file_name_list[-3] == '100X':
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_bengin_test))
                process.start()
            if file_name_list[-3] == '200X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\200X\bengin')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_bengin_test))
                process.start()
            if file_name_list[-3] == '400X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\400X\bengin')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_bengin_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_bengin_test))
                process.start()
                continue
        elif file_name_list[-1] == 'maligant':
            if file_name_list[-3] == '40X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\40X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_40_maligant_test))
                process.start()
            if file_name_list[-3] == '100X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\100X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_100_maligant_test))
                process.start()
            if file_name_list[-3] == '200X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\200X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_maligant_train))
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_200_maligant_test))
                process.start()
            if file_name_list[-3] == '400X':
                # split_all_img(root, r'D:\Breast-cancer-detection\code\data-process\image-quartering\400X\maligant')
                if file_name_list[-2] == "train":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_maligant_train))
                    process.start()
                elif file_name_list[-2] == "test":
                    process = multiprocessing.Process(target=split_all_img, args=(root, target_path_400_maligant_test))
                    process.start()

    print('十六等分完成')


