# augmentation script

import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

def augment_data(class_dir, target_class_dir, augmentations, num_augmented_images):
        datagen = ImageDataGenerator(**augmentations)

        images = [img for img in os.listdir(class_dir) if img.endswith(('png', 'jpg', 'jpeg'))]

        num_augmented_images = num_augmented_images // len(images)

        for img_name in images:
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate augmented images
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=target_class_dir, save_prefix='aug', save_format='jpeg'):
                i += 1
                if i >= num_augmented_images:
                    break

augmentations = {
    'rotation_range': 40,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

source_dir = '/content/drive/MyDrive/resnet50/dataset/train/'
target_dir = '/content/drive/MyDrive/resnet50/dataset/train/'

class_counts_df = pd.read_csv('class_names_with_count.csv')

# Augment data for each class
for _, row in class_counts_df.iterrows():
    class_name = row['class']
    count = row['count']
    # Define the number of augmented images needed to balance the dataset
    num_augmented_images = 100 - count  # Assume max_count is the target number of images per class
    # print('aaaa')
    augment_data(os.path.join(source_dir, class_name), os.path.join(target_dir, class_name), augmentations, num_augmented_images)
