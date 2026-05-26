import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128

def load_images(data_dir):
    images = []
    labels = []
    classes = os.listdir(data_dir)

    for label, disease in enumerate(classes):
        disease_path = os.path.join(data_dir, disease)
        for img in os.listdir(disease_path):
            try:
                img_path = os.path.join(disease_path, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                images.append(image)
                labels.append(label)
            except:
                pass

    images = np.array(images) / 255.0
    labels = to_categorical(labels, num_classes=len(classes))

    return images, labels, classes
