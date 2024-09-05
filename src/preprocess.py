import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore

# Preprocess the image
def load_and_process_image(img_path, target_size):
    img = image.load_img(img_path, target_size = target_size) 
    img_ary = image.img_to_array(img)
    img_ary = tf.expand_dims(img_ary, axis = 0)
    img_ary = img_ary / 255.0
    return img_ary