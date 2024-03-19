import os
import shutil

def FindFile(filePath, newFilePath):
    filename = os.listdir(filePath)
    for i in filename:
        shutil.copy(filePath + '/' + i, newFilePath + '/' + i)
def file_name_walk(file_dir, benginOrmalignant):
    for root, dirs, files in os.walk(file_dir):
        if benginOrmalignant == 0:
            # print("benign")
            if root.split('\\')[-1] == '40X':
                FindFile(root, r'.\data-process\bengin-and-maligant\40X\bengin')
            if root.split('\\')[-1] == '100X':
                FindFile(root,r'.\data-process\bengin-and-maligant\100X\bengin')
            if root.split('\\')[-1] == '200X':
                FindFile(root, r'.\data-process\bengin-and-maligant\200X\bengin')
            if root.split('\\')[-1] == '400X':
                FindFile(root, r'.\data-process\bengin-and-maligant\400X\bengin')
        elif benginOrmalignant == 1:
            # print("malignant")
            if root.split('\\')[-1] == '40X':
                FindFile(root, r'.\data-process\bengin-and-maligant\40X\maligant')
            if root.split('\\')[-1] == '100X':
                FindFile(root, r'.\data-process\bengin-and-maligant\100X\maligant')
            if root.split('\\')[-1] == '200X':
                FindFile(root, r'.\data-process\bengin-and-maligant\200X\maligant')
            if root.split('\\')[-1] == '400X':
                FindFile(root, r'.\data-process\bengin-and-maligant\400X\maligant')

#Benign Pathology Pictures Folder Path
file_name_walk(r'.\data-process\BreaKHis_v1\histology_slides\breast\benign\SOB', 0)
#Malignant Pathology Picture Folder Path
file_name_walk(r'.\data-process\BreaKHis_v1\histology_slides\breast\malignant\SOB', 1)
