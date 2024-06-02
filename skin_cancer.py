#****************************#phase 1************************#

import os
import cv2
import numpy as np
from tensorflow.keras.layers import MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM
from tensorflow.keras.models import Sequential
from keras_preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

dataset_path = "C:/Users/asims/Desktop/hackathon/skin_cancer_8"  
image_width = 128
image_height = 128
num_channels = 3  
num_classes = len(os.listdir(dataset_path))

def load_dataset(dataset_path, image_width, image_height):
    images = []  
    labels = []  
    label_encoder = LabelEncoder()

    for class_name in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_name)
        for image_file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_file)
           
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_width, image_height))
           
            images.append(image)
            labels.append(class_name)

    labels = label_encoder.fit_transform(labels)

    return np.array(images), to_categorical(labels, num_classes=num_classes)

images, labels = load_dataset(dataset_path, image_width, image_height)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

from keras.layers import Reshape 

model = Sequential()

model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(image_width, image_height, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

num_frames = 10 
model.add(Reshape((num_frames, -1)))

model.add(LSTM(512, return_sequences=True))
model.add(LSTM(256, return_sequences=False))

model.add(Dense(512, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")


#****************************#testing************************#

from keras.models import load_model
model = load_model('skin_cancer_new.h5')

import numpy as np
from keras.preprocessing import image

img_path = r"C:\Users\asims\Desktop\img\segmented_imagemel.png"
img = image.load_img(img_path, target_size=(128, 128))  
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0  
predictions = model.predict(img)

predicted_class_index = np.argmax(predictions)
print("Predicted Class Index:", predicted_class_index)

class_names = ['Normal','atypical moles','benigen_keratosis','lentigo maligma melanoma','malignant','melanocytic_nevi','melanoma','vascular']
predicted_class_name = class_names[predicted_class_index]
print("Predicted Class:", predicted_class_name)
