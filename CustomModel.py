import tensorflow as tf
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv_layer1    = Conv2D(filters=32, kernel_size=(5,5), activation="relu")
        self.conv_layer2    = Conv2D(filters=32, kernel_size=(5,5), activation="relu")
        self.max_pool1      = MaxPool2D(pool_size=(2,2))
        self.dropout_layer1 = Dropout(rate=0.25)
        self.conv_layer3    = Conv2D(filters=64, kernel_size=(3,3), activation="relu")
        self.conv_layer4    = Conv2D(filters=64, kernel_size=(3,3), activation="relu")
        self.max_pool2      = MaxPool2D(pool_size=(2,2))
        self.dropout_layer2 = Dropout(rate=0.5)
        self.flatten        = Flatten()
        self.dense_layer1   = Dense(256, activation="relu")
        self.dropout_layer3 = Dropout(rate=0.5)
        self.dense_layer2   = Dense(43, activation="softmax")

    def build(self, input_shape):
        input_tensor = tf.keras.Input(shape=input_shape[1:])
        self.call(input_tensor)
        super(CustomModel, self).build(input_shape)

    def call(self, inputs):
        x = self.conv_layer1(inputs)
        x = self.conv_layer2(x)
        x = self.max_pool1(x)
        x = self.dropout_layer1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.max_pool2(x)
        x = self.dropout_layer2(x)
        x = self.flatten(x)
        x = self.dense_layer1(x)
        x = self.dropout_layer3(x)
        x = self.dense_layer2(x)
        return x