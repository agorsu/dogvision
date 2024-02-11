from PIL import ImageOps, Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

#load class names
with open('model/labels.csv', 'r') as f:
    class_names = [a[:-1].split(',')[1] for a in f.readlines()]
    class_names = class_names[1:]
    f.close()
print(f'{len(class_names)} class names')

# Find the unique label values
unique_breeds = np.unique(class_names)
print(f'{len(unique_breeds)} breeds')

def get_pred_label(prediction_probabilities):
  return unique_breeds[np.argmax(prediction_probabilities)]

def classify(image, model):
    """
    This function takes an image and a model and returns the predicted class and confidence
    score of the image.

    Parameters:
        image: Dog image to be classified.
        model: A trained machine learning model for image classification.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    class_name = get_pred_label(prediction)
    confidence_score = np.max(prediction)*100

    return class_name, confidence_score

def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model