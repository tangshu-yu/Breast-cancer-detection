import re
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import Callback
from keras import optimizers
# Training set data path
train_data_dir = r'.\train_test_data\sixteen\40X\train'
img_width, img_height = 299, 299
batch_size = 32

class LossHistory(Callback):
    def __init__(self):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        loss = logs.get('loss')
        self.losses.append(loss)

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
predictions = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
adam = optimizers.adam_v2
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
loss_and_acc_history = model.fit(
    train_generator,
    epochs=100
)
dire_list = re.split(r'[\\]',train_data_dir)
model_name = 'result\\train_'+dire_list[-3]+'_'+dire_list[-2]+'.h5'
model.save(model_name)
