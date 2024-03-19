Environmental requirements:

tensorflow == 2.6.0

python == 3.7

[image-route-process.py](image-route-process.py) Classifying images in the breakhis dataset into benign and malignant categories

[data_split.py](data_split.py) Divide the data into training and test sets

[image-quartering-sixteen.py](image-quartering-sixteen.py) Split the image into four and sixteen pieces of the same size

[model.py](model.py) training model

[model_prediction.py](model_prediction.py) Evaluation Models

First you need to use [image-route-process.py](image-route-process.py) to divide the BreakHis dataset 
into benign and malignant categories, then use [data_split.py](data_split.py) to divide the 
dataset into training and test sets, 
and thirdly use [image-quartering-sixteen.py](image-quartering-sixteen.py) to segment the 
images into fourth and sixteenth class segmentation, fourth use [model.py](model.py) to train 
the model and finally use [model_prediction.py](model_prediction.py)  to obtain the 
evaluation metrics of the model.