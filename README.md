# NAIVEBAYES
THESIS Algorithm
The code begins by loading water quality data from a CSV file, which includes information about pH, turbidity, and water cleanliness labels.

It then prepares the data by extracting the features (pH and turbidity) and the target variable (cleanliness labels) and performs data augmentation by duplicating some of the data for better training.

The data is split into training and testing sets, and feature scaling is applied using standardization.

A neural network model is defined with two layers, one with ReLU activation and another with sigmoid activation, suitable for binary classification tasks.

The model is compiled with the Adam optimizer and binary cross-entropy loss, and accuracy is used as the evaluation metric.

The model is trained for 200 epochs using the training data.

Finally, the trained model is converted into a TensorFlow Lite format for deployment on resource-constrained platforms, such as mobile devices and embedded systems.
