from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, MaxPool2D, \
    Lambda, AveragePooling2D, TimeDistributed, ConvLSTM2D, Reshape, SpatialDropout2D, SeparableConv2D
from tensorflow.keras import regularizers, Model
from tensorflow.keras.constraints import max_norm

import numpy as np

def create_model():    
    learning_rate = 5e-05 # typically a small number for adam optimizer
    
    model = Sequential()
    model.add(InputLayer(input_shape=(21,24,1))) # input layer of size (num time windows) * (num csp) * 1(num_channels)

    name = 'conv_layer_1'
    model.add(Conv2D(kernel_size=[2, 4], 
                                    strides=[1, 4], filters=16, 
                                    padding='same',  activation='relu', 
                                    name=name))
    
    model.add(MaxPooling2D(pool_size=(1, 3), strides=(1, 3)))  # Adjust pool_size and strides as needed
    
    name = 'conv_layer_2'
    model.add(Conv2D(kernel_size=[2, 4], 
                                    strides=[1, 4], filters=32, 
                                    padding='same',  activation='relu', 
                                    name=name))

    # then, enter dense layers
    model.add(Flatten())

    name = 'layer_dense_1'
    model.add(Dense(16, activation='relu', name=name)) 
        
    #name = 'layer_dense_2'
    #model.add(Dense(8, activation='relu', name=name)) 
    
    name = 'layer_output'
    model.add(Dense(2, activation='softmax', name=name)) 

    optimizer = Adam(lr=learning_rate)

    model.compile(optimizer=optimizer,
          loss='categorical_crossentropy',
          metrics=['accuracy'])
    
    return model


# the online model
def EEGNet(nb_classes, Chans=8, Samples=250,
           dropoutRate=0.5, kernLength=125, F1=7,
           D=2, F2=7, norm_rate=0.25, dropoutType='Dropout'):
    """ tensorflow.Keras Implementation of EEGNet

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense', kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)



def train_model(csp_train, y_train, csp_val, y_val, model, batch_size, epochs):
    hist = model.fit(x=csp_train,
              y=y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(csp_val, y_val))
    
    return hist


# train with plotting
def fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size):
    # fits the network epoch by epoch and saves only accurate models

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(epochs):
        history = model.fit(train_X, train_y, epochs=1, batch_size=batch_size,
                            validation_data=(validation_X, validation_y))

        train_acc.append(history.history["accuracy"][-1])
        train_loss.append(history.history["loss"][-1])
        val_acc.append(history.history["val_accuracy"][-1])
        val_loss.append(history.history["val_loss"][-1])

        '''MODEL_NAME = f"models/{round(val_acc[-1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(val_loss[-1], 2)}.model"

        if round(val_acc[-1] * 100, 4) >= 77 and round(train_acc[-1] * 100, 4) >= 77:
            # saving & plotting only relevant models
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)

            # Accuracy
            plt.plot(np.arange(len(val_acc)), val_acc)
            plt.plot(np.arange(len(train_acc)), train_acc)
            plt.title('Model Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['val', 'train'], loc='upper left')
            plt.show()

            # Loss
            plt.plot(np.arange(len(val_loss)), val_loss)
            plt.plot(np.arange(len(train_loss)), train_loss)
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('epoch')
            plt.legend(['val', 'train'], loc='upper left')
            plt.show()'''
    
def check_accuracy(model, X, y_truth, y_pred):
    correct = 0
    for i in range(len(y_truth)):
        if (y_pred[i] == y_truth[i]):
            correct += 1
            
    return correct/len(y_truth)

def model_evaluation(model, X, y_truth):
    y_pred = np.argmax(model.predict(X), axis = 1)
    y_truth = np.argmax(y_truth, axis = 1)
    
    test_accuracy = check_accuracy(model, X, y_truth, y_pred)
    
    print('test accuracy: {}'.format(test_accuracy))
    
    return test_accuracy
        