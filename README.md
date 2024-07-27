# Image-Classification-using-Deep-Learning

Dataset
The dataset used for this project is the popular Kaggle Dogs vs. Cats dataset. It consists of 25,000 labeled images of dogs and cats for training and 12,500 images for testing.

Model Architecture
The CNN model consists of the following layers:
Convolutional Layer: Applies convolution operations to the input image.
Activation Layer: Uses ReLU activation function.
Pooling Layer: Performs max pooling to reduce the spatial dimensions.
Dropout Layer: Applies dropout for regularization.
Flatten Layer: Flattens the pooled feature maps into a single vector.
Dense Layer: Fully connected layer with a ReLU activation function.
Output Layer: Fully connected layer with a sigmoid activation function for binary classification.

Training:
The model was trained using the binary cross-entropy loss function and the Adam optimizer. The training process involved the following steps:
Preprocessing the data: Resizing and normalizing the images.
Defining the model architecture.
Compiling the model.
Training the model with the training data and validating it with validation data.

Evaluation:
The model was evaluated on the test dataset to measure its accuracy. The accuracy achieved by the model was calculated after training.


Results:
The model was able to accurately classify images of dogs and cats. The performance metrics, such as accuracy, precision, recall, and F1-score, can be included here based on the evaluation results.

Conclusion:
This project successfully implemented a CNN for image classification to distinguish between images of dogs and cats. The model demonstrated good performance, indicating the effectiveness of CNNs for image classification tasks. Future improvements could involve experimenting with more complex architectures, data augmentation techniques, and transfer learning.
