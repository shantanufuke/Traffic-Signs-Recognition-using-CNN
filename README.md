# Traffic-Signs-Recognition-using-CNN

![image](https://github.com/shantanufuke/Traffic-Signs-Recognition-using-CNN/assets/104629474/5078a09d-3688-4d6f-9241-733f8acc19b2)

I have build a model for the classification of traffic signs available in the image into many categories using a convolutional neural network(CNN) and the Keras library.

## Dataset for Traffic Sign Recognition**

The image dataset consists of more than 50,000 pictures of various traffic signs(speed limit, crossing, traffic signals, etc.) Around 43 different classes are present in the dataset for image classification. It contains two separate folders, train and test, where the train folder consists of classes, and every category contains various images.

![image](https://github.com/shantanufuke/Traffic-Signs-Recognition-using-CNN/assets/104629474/dfea8ac8-5e7f-40fb-8ff9-9e7785662c3d)

## You can download the Kaggle dataset for this project from the below link.**
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Workflow:

## Dataset Exploration:

Explore the dataset containing traffic sign images organized into subfolders representing different classes.
Utilize the OS module to iterate through the images and their labels, and the PIL library to open and process image data.
Store image data and labels into lists and convert them into NumPy arrays for model training.

## CNN Model Building:

Implement a custom CNN architecture using TensorFlow's Keras API, defining convolutional layers with ReLU activation, max-pooling, and dropout layers to prevent overfitting.
Configure convolutional layers with varying filter sizes and dropout rates to capture hierarchical features from input images.
Add dense layers with ReLU activation and a final output layer with softmax activation to yield class probabilities.
Construct the model to process input images of specified shape, facilitating effective image classification.

## Model Training and Validation:

Train the model using the model.fit() method, specifying batch size and epochs to optimize performance.
Evaluate model performance on training and validation sets, monitoring accuracy and stability to ensure effective learning.

## Model Testing:

Utilize a separate test dataset containing image paths and corresponding class labels.
Resize test images to match the model input dimensions and generate a NumPy array with image data.
Predict class labels using the trained model and evaluate predictions using accuracy_score from sklearn.metrics.

## GUI for Traffic Signs Classifier:

Use Gradio, a Python library for building customizable UI components for machine learning models, to create a graphical user interface.
Build a GUI interface with radio buttons to upload images and a classifier button for prediction.
Implement a classify() function to process uploaded images, predict traffic sign classes, and display results in a visually appealing format.

## Conclusion:
We successfully developed a CNN model to classify traffic signs with 95% accuracy, leveraging a large dataset and implementing a custom architecture. The intuitive GUI built with Gradio enhances user interaction and understanding of the classification process, providing a seamless experience for traffic sign identification. 
