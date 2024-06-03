# Traffic-Signs-Recognition-using-CNN

![image](https://github.com/shantanufuke/Traffic-Signs-Recognition-using-CNN/assets/104629474/5078a09d-3688-4d6f-9241-733f8acc19b2)

We will build a model for the classification of traffic signs available in the image into many categories using a convolutional neural network(CNN) and Keras library.

**Dataset for Traffic Sign Recognition**

The image dataset consists of more than 50,000 pictures of various traffic signs(speed limit, crossing, traffic signals, etc.) Around 43 different classes are present in the dataset for image classification. It contains two separate folders, train and test, where the train folder consists of classes, and every category contains various images.

![image](https://github.com/shantanufuke/Traffic-Signs-Recognition-using-CNN/assets/104629474/dfea8ac8-5e7f-40fb-8ff9-9e7785662c3d)

**You can download the Kaggle dataset for this project from the below link.**
https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

**Workflow**
We need to follow the below 4 steps to build our traffic sign classification model:
  CNN model building
  Model training and validation
  Model testing

**Dataset exploration**
Around 43 subfolders(ranging from 0 to 42) are available in our ‘train’ folder, and each subfolder represents a different class. We have an OS module that helps in the iteration of all the images with their respective classes and labels. To open the contents of ideas into an array, we are using the PIL library.
In the end, we have to store every image with its corresponding labels into lists. A NumPy array is needed to feed the data to the model, so we convert this list into an array. Now, the shape of our data is (39209, 30, 30, 3), where 39209 represents the number of images, 30*30 represents the image sizes into pixels, and the last 3 represents the RGB value(availability of coloured data).

**CNN model building**
The CustomModel class defines a convolutional neural network architecture for image classification tasks using TensorFlow's Keras API. It comprises two sets of convolutional layers with ReLU activation followed by max-pooling and dropout layers to prevent overfitting. Each set consists of convolutional layers with varying filter sizes and dropout rates. After the convolutional layers, the network flattens the output and passes it through two dense layers with ReLU activation before the final output layer with softmax activation, yielding class probabilities. The model is built to process input images of specified shape, capturing hierarchical features through convolutional layers and making predictions through fully connected layers, thus serving as an effective tool for image classification tasks.

**Model training and validation**
To train our model, we will use the model.fit() method that works well after the successful building of model architecture. With the help of 64 batch sizes, we got 95%accuracy on training sets and acquired stability after 15 epochs.

**Model testing**
A folder named” test” is available in our dataset; inside that, we got the main working comma-separated file called” test.csv”. It comprises two things, the image paths, and their respective class labels. We can use the pandas’ Python library to extract the image path with corresponding labels. Next, we need to resize our images to 30×30 pixels to predict the model and create a numpy array filled with image data. To understand how the model predicts the actual labels, we need to import accuracy_score from the sklearn.metrics.

**GUI for Traffic Signs Classifier**
We will use a standard Python library called Tkinter to build a graphical user interface(GUI) for our traffic signs recognizer. We need to create a separate python file named” gui.py” for this purpose. Firstly, we need to load our trained model ‘traffic_classifier.h5’ with the Keras library’s help of the deep learning technique. After that, we build the GUI to upload images and a classifier button to determine which class our image belongs to. We create a classify() function for this purpose; whence we click on the GUI button, this function is called implicitly. To predict the traffic sign, we need to provide the same resolutions of shape we used at the model training time. So, in the classify() method, we convert the image into the dimension of shape (1 * 30 * 30 * 3). The model.predict_classes(image) function is used for image prediction, it returns the class number(0-42) for every image. Then, we can extract the information from the dictionary using this class number.

**Conclusion**
We created a CNN model to identify traffic signs and classify them with 95% accuracy. We had observed the accuracy and loss changes over a large dataset. GUI of this model makes it easy to understand how signs are classified into several classes.

