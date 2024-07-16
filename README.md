# Retinal-Disease-Detection-using-Deep-Learning-Networks
Overview
This project trains a Convolutional Neural Network (CNN) to classify medical images from the OCT2017 dataset into four categories: CNV, DME, DRUSEN, and NORMAL. The dataset is split into training, testing, and validation sets.

Dataset
The dataset consists of images of size 128x128 pixels, categorized into four classes:

CNV
DME
DRUSEN
NORMAL
The dataset is divided into three directories:

Training data: trainnew
Testing data: testnew
Validation data: validation
Preprocessing
Image Normalization: All images are normalized to have pixel values in the range [0, 1] by rescaling with a factor of 1./255.
Data Generators: ImageDataGenerator is used to generate batches of image data with real-time data augmentation.
Model Architecture
The CNN model is built using TensorFlow and Keras. The architecture is as follows:

Convolutional layers with ReLU activation
Max pooling layers
Flatten layer
Dropout layer for regularization
Fully connected (Dense) layers
Output layer with softmax activation for multi-class classification

Prediction
The model is used to predict the category of a given test image. The steps are as follows:

Load and preprocess the image.
Make a prediction using the loaded model.
Print the predicted category.

Model Evaluation
The model's performance is evaluated on the test set:

Load and preprocess the test data.
Predict labels for the test data using the model.
Calculate accuracy by comparing predicted labels with actual labels.

Conclusion
This project demonstrates the process of training, evaluating, and using a Convolutional Neural Network for image classification. The model achieves high accuracy on the test set, indicating its effectiveness in classifying medical images into the four specified categories.







