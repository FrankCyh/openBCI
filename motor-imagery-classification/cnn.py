import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_model(hyperparams):
    learning_rate, conv_size_1, conv_size_2, \
        num_conv_filters, num_conv_layers, \
        num_dense_nodes, num_dense_layers = hyperparams


    model = Sequential()
    model.add(InputLayer(input_shape=(11, 36, 1)))

    for i in range(int(num_conv_layers)):
        name = 'conv_layer_{0}'.format(i + 1)

        model.add(Conv2D(
            kernel_size=[conv_size_1, conv_size_2],
            strides=[1, 6],
            filters=num_conv_filters,
            padding='same',
            activation='relu',
            name=name,
        ))

    model.add(Flatten())

    for i in range(int(num_dense_layers)):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(
            int(num_dense_nodes),
            activation='relu',
            name=name,
        ))

    model.add(Dense(2, activation='softmax'))

    optimizer = Adam(lr=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def train_model(csp_train, y_train, csp_val, y_val, model, batch_size, epochs):
    hist = model.fit(
        x=csp_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(csp_val, y_val),
    )

    return hist


def check_accuracy(model, X, y_truth, y_pred):
    correct = 0
    for i in range(len(y_truth)):
        if (y_pred[i] == y_truth[i]):
            correct += 1

    return correct / len(y_truth)


def model_evaluation(model, X, y_truth):
    y_pred = np.argmax(model.predict(X), axis=1)
    y_truth = np.argmax(y_truth, axis=1)

    test_accuracy = check_accuracy(model, X, y_truth, y_pred)

    print('test accuracy: {}'.format(test_accuracy))

    return test_accuracy
