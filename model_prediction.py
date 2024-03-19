from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import re
import os
os.environ["TFHUB_CACHE_DIR"] = "some_dir"


def calculate_precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)

def calculate_recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

def calculate_f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

def calculate_specificity(true_negative, false_positive):
    return true_negative / (true_negative + false_positive)

def calculate_sensitivity(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

def matthews_correlation_coefficient(TP, TN, FP, FN):
    numerator = TP * TN - FP * FN
    denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    if denominator == 0:
        return 0  # To handle division by zero
    else:
        return numerator / denominator


input_shape  = (299, 299, 3)
#Test Data Catalog
test_data_dir = r'...'
# Reload the model
model_path = r'...'
model = load_model(model_path)
print(model.summary())
dir_list = re.split(r'[\\]', test_data_dir)
# 1 for no division, 4 for 4 divisions, 16 for 16 divisions
if dir_list[-3] == 'sixteen':
    partition_size = 16
elif dir_list[-3] == 'quartering':
    partition_size = 4
else:
    partition_size = 1

# Make projections
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_generator, verbose=1)
label_true = test_generator.classes.reshape(1, test_generator.classes.shape[0])
label_true = np.squeeze(label_true)
file_name_list = test_generator.filenames.copy()
complete_image_num = 0
TP = 0
FN = 0
FP = 0
TN = 0
# Total number of pictures
image_num = 0
# correctly categorized based on images
image_true_num = 0
while len(file_name_list) > 0:
    # Split file names at the image level
    image_name = re.split(r'[_\\]', file_name_list[0])
    search_image_string = ''.join(image_name[1]) + ('_') + (image_name[2]) + ('_') + (image_name[3])
    image_indices_index = [index for index, value in enumerate(test_generator.filenames) if
                           search_image_string.lower() in value.lower()]
    image_delete_index = [index for index, value in enumerate(file_name_list) if
                          search_image_string.lower() in value.lower()]
    if image_indices_index:
        # Get the prediction of the segmented image
        pre = predictions[image_indices_index]
        pre_sum = np.sum(pre, axis=0) / partition_size
        pre_label = np.argmax(pre_sum, axis=0)
        # To get the true value
        label = label_true[image_indices_index]
        if not np.all(label == label[0]):
            print("tagging error")
            break
        if label[0] == pre_label:
            image_true_num = image_true_num + 1
            if label[0] == 0:
                TP = TP + 1
            elif label[0] == 1:
                TN = TN + 1
        elif label[0] != pre_label:
            if label[0] == 0:
                FP = FP + 1
            elif label[0] == 1:
                FN = FN + 1
        # Delete matching items
        for index in reversed(image_delete_index):
            del file_name_list[index]
    else:
        print(f"No fuzzy matches '{search_image_string}' The items in the list of")
        break
    complete_image_num = complete_image_num + 1

    image_num = image_num + len(image_delete_index)
print("Total number of images：%d" % image_num)
print("Number of complete pictures：%d" % complete_image_num)
print("TP={},TN={},FP={},FN={}" .format(TP,TN,FP,FN) )
image_acc = (TP+TN) / (TP+TN+FN+FP)
print("Accuracy:%.5f" % image_acc)
Precision = (TP)/(TP+FP)
Recall = (TP)/(TP+FN)
F1_Score = 2 * ((Precision * Recall) / (Precision + Recall))
print("F1_Score:%.5f" % F1_Score)
Specificity = TN / (TN + FP)
print("Specificity:%.5f" % Specificity)
Sensitivity = TP / (TP + FN)
print("Sensitivity:%.5f" % Sensitivity)
mcc = matthews_correlation_coefficient(TP, TN, FP, FN)
print("MCC:", mcc)
