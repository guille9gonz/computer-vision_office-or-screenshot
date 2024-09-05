from constants import IMAGE_SIZE
from preprocess import load_and_process_image
from cnn_model import CNN

def main(image_path):
    preprocessed_image = load_and_process_image(image_path, IMAGE_SIZE)

    cnn_model = CNN()

    predictions = cnn_model.predict(preprocessed_image)

    result = cnn_model.prediction_output(predictions)

    cnn_model.image_plot(image_path, result)

# Main gets the image path and starts the prediction
# Path has to use "/" instead of "\"
if __name__ == '__main__':
    image_path = "testing\img_office_1.png"
    main(image_path)