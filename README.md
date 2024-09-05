# Project Title: Image Classification with Convolutional Neural Network (CNN)
## Overview
This project implements a Convolutional Neural Network (CNN) model to classify images into two categories: Office and Screenshot. The model is trained using TensorFlow and Keras on a custom dataset of 1,600 images, categorized as "office" or "screenshot".

The project includes Python scripts for data preprocessing, model training, and evaluation, as well as a Jupyter Notebook that details the full process from data preparation to model evaluation.

## Project Structure
The repository contains the following files:
  - **`cnn_model/`**:
    - `training_model_cnn.ipynb`: A Jupyter Notebook with all the steps to load the data, preprocess, model architecture, training and evaluation.
    - `my_cnn_model.keras`: The trained model saved in a Keras file.
    
  - **`src/`**:
    - `constants.py`: Contains the constants of the input image size and path to the model.
    - `preprocessing.py`: Handles the input image preprocessing, transforming it into an array and normalizing to [0, 1].
    - `cnn_model.py`: Contains a class with the constructor to load the model and functions to do the prediction and output it.
    - `main.py`: Script for executing the model.

  - **`requirements.txt`**: Lists the dependencies required to run the model properly.
  -  **`.gitignore`**: Specifies which files and directories should be ignored by Git (virtual environment, dataset and cache).
   
## Results
The model used a total of 1,623 images belongin to 2 classes: 756 for "office" and 867 for "screenshot". The dataset was split 80% for training and 20% for validation.
It uses 3 convolutional blocks, each containing 3 layers, and a classifier head with 4 additional layers. The model is trained for 10 epochs.
  - **Training Accuracy**: 99.5%
  - **Validation Accuracy**: 99.1%

The model performs with high accuracy, even with ambiguous images (e.g., a screenshot displaying an office image — it correctly identifies it as a screenshot).
