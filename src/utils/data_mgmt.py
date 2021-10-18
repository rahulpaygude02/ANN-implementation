import tensorflow as tf

def get_data(validation_datasize):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    #creating validation dataset from training data
    #scale the data bet 0 & 1 by dividing it by 255.

    x_valid,x_train_small = x_train[:validation_datasize]/255. , x_train[validation_datasize:]/255.
    y_valid,y_train_small = y_train[:validation_datasize] , y_train[validation_datasize:]

    x_test = x_test/255.

    return (x_train_small,y_train_small),(x_valid,y_valid), (x_test,y_test)