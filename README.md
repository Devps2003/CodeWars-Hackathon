

# Video Classification using Convolutional Neural Networks

This project aims to classify videos into real or fake categories using Convolutional Neural Networks (CNNs). The dataset consists of videos obtained from various sources, including Celeb-real, Youtube-real, Celeb-Youtube-fake, and a test set.

## This is the model file drive link
https://drive.google.com/file/d/1vOGZ2kVkyfVLU0sFIZydaY7rLky3Toh9/view?usp=sharing

## Disclaimer
To accommodate our system limitations, we created a reduced dataset from the original provided dataset, as processing the entire 10GB dataset proved impractical and time-consuming. Therefore, our submission and scores are based on this smaller dataset.

## Models and Techniques Used

1. **Data Preprocessing**: Videos are preprocessed by extracting frames and saving them as images. Each frame is resized to 224x224 pixels and normalized to have pixel values in the range [0, 1].

2. **Data Augmentation**: Data augmentation techniques such as rotation, flipping, and zooming were not explicitly applied in this project due to the nature of video classification, where the temporal sequence of frames already captures variations.

3. **Convolutional Neural Network (CNN)**: A CNN architecture is used for classification. It consists of three convolutional layers followed by max-pooling layers for feature extraction, a flatten layer to convert the 2D feature maps into a 1D vector, and two fully connected (dense) layers with ReLU activation for classification. The output layer has a sigmoid activation function for binary classification.

4. **Binary Crossentropy Loss**: Binary crossentropy loss is used as the loss function, which is suitable for binary classification tasks.

5. **Adam Optimizer**: The Adam optimizer is used for training the model.

6. **Evaluation Metrics**: The model is evaluated using accuracy, precision, recall, and F1-score metrics.

## Instructions to Run on Google Colab

To run the model on Google Colab, follow these steps:

1. **Upload Dataset**: Upload the dataset folders ("Celeb-real", "Youtube-real", "Celeb-Youtube-fake", "test") to your Google Drive.

2. **Mount Google Drive**: Mount your Google Drive in Google Colab using the following code:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Navigate to Dataset Directory**: Navigate to the directory containing the dataset folders using the following command:
    ```python
    %cd /content/drive/MyDrive/path_to_dataset_folders
    ```

4. **Clone Repository**: Clone the repository containing the model code and README:
    ```bash
    !git clone https://github.com/your_username/your_repository.git
    ```

5. **Run the Model Code**: Open the notebook or Python script containing the model code and run it.

6. **View Results**: Once the model has finished training and evaluation, you can view the results such as accuracy, precision, recall, and F1-score.

7. **Download Outputs**: You can download the outputs such as the predictions CSV file, accuracy plot, and F1-score PDF from the Colab environment to your local machine.

## Results

After running the model, you can find the following outputs:

- **submission.csv**: CSV file containing the filenames, true labels, and predicted labels for the test dataset.
- **accuracy_plot.pdf**: PDF file containing the plot of training accuracy and validation accuracy over epochs.
- **f1_score.pdf**: PDF file containing the F1-score, precision, and recall metrics on the test dataset.
