import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import matplotlib.pyplot as plt



# Load the  model
model = tf.keras.models.load_model('models/test_case2_classification.h5')
# classes.txt file path
class_names_path = 'lists/lists/classes.txt'


with open(class_names_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(image_path):
    """
    Predict the class of a given image.
    
    Args:
    image_path (str): Path to the input image.
    
    Returns:
    tuple: Predicted class name and confidence score.
    """

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Make prediction
    predictions = model.predict(img_array)    
    # Get the predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    class_name = class_names[predicted_class]
    confidence = predictions[0][predicted_class]
    return class_name, confidence

def classify_and_display_image(image_path):
    """
    Classify a single image, display it, and print the result.
    
    Args:
    image_path (str): Path to the input image.
    """

    class_name, confidence = predict_image(image_path)
    #Take only class name
    class_name = class_name.split('.')[1]  
    # Load and display the image
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    # Add prediction information on the image
    plt.title(f"Predicted: {class_name}\nConfidence: {confidence*100:.2f}%")
    plt.show()
    
    print(f"Uploaded image: {os.path.basename(image_path)}")
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence*100:.2f}%")

def classify_images_in_directory(directory):
    """
    Classify all images in a given directory.
    
    Args:
    directory (str): Path to the directory containing images.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(directory, filename)
            classify_and_display_image(image_path)
            print("---")

# for single image classification
print("Single Image Classification:")
single_image_path = 'testing_images/parakeetAuklet.jpg'
classify_and_display_image(single_image_path)
# classify images in a directory
directory = 'testing_images'
classify_images_in_directory(directory)