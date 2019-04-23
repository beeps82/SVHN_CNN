import numpy as np
import os
import helper
import detection
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras import regularizers
import keras.utils as ku
from keras import backend as K
from keras import optimizers
import tensorflow as tf
import pickle
import matplotlib.pyplot as ply

# Includes training models: Designed, VGG-16, VGG-16 Pre Trained 
# & Metrics and plots to measure performance
#
#
# I/O directories
test_dir = "test"
train_dir = "train"
OUTPUT_DIR = "output"

train = os.path.join(train_dir, "train")
test  = os.path.join(test_dir, "test")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)


def designedCNN_Model():
    data = helper.prepDataforCNN(numChannel = 3, feat_norm = True)
    trainX = data["trainX"]
    valdX  = data["valdX"]
    trainY = data["trainY"]
    valdY  = data["valdY"]

    _,row, col,channel = trainX.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 75
    batch_size = 64

    optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    # optim = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config = config)

    input = keras.Input(shape=(row,col,channel), name='customModel')
    M = Conv2D(16,(3,3),activation='relu',padding='same',name = 'conv_16_1')(input)
    M = Conv2D(16,(3, 3), activation ='relu', padding='same',name = 'conv_16_2')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)

    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_32_01')(M)
    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_32_02')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)
    M = Dropout(0.5)(M)

    M = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv2_48_01')(M)
    M = Conv2D(48, (3, 3), activation ='relu', padding='same', name = 'conv2_48_02')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)

    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_64_1')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same', name = 'conv2_64_2')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_64_3')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)

    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_1')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_2')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_3')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)

    M = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_5')(M)
    M = Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv2_128_6')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)
    M = Dropout(0.5)(M)

    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_1')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_2')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'conv256_3')(M)
                # kernel_regularizer=regularizers.l2(0.01),
                # activity_regularizer=regularizers.l1(0.01))(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)

    M = Conv2D(512, (5, 5), activation='relu', padding='same',name = 'conv2_512_1')(M)
    M = Conv2D(512, (5, 5), activation='relu', padding='same',name = 'conv2_512_2')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides= 1)(M)
    M = Dropout(0.25)(M)
    # M = keras.layers.BatchNormalization(axis=-1)(M)

    Mout = Flatten()(M)
    Mout = Dense(2048, activation='relu', name = 'FC1_2048')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC1_1024')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC2_1024')(Mout)
    # Mout = Dropout(0.5)(Mout)

    numd_SM = Dense(digLen,    activation='softmax',name = 'num')(Mout)
    dig1_SM = Dense(numDigits, activation='softmax',name = 'dig1')(Mout)
    dig2_SM = Dense(numDigits, activation='softmax',name = 'dig2')(Mout)
    dig3_SM = Dense(numDigits, activation='softmax',name = 'dig3')(Mout)
    dig4_SM = Dense(numDigits, activation='softmax',name = 'dig4')(Mout)
    numB_SM = Dense(2,         activation='softmax',name = 'nC')(Mout)
    out = [numd_SM, dig1_SM ,dig2_SM, dig3_SM, dig4_SM, numB_SM]

    svhnModel = keras.Model(inputs = input, outputs = out)

    lr_metric = get_lr_metric(optim)
    svhnModel.compile(loss = 'sparse_categorical_crossentropy', #ceLoss ,
                      optimizer= optim,
                      metrics=  ['accuracy']) #[])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                  factor = 0.1,
                                                  verbose = 1,
                                                  patience= 2,
                                                  cooldown= 1,
                                                  min_lr = 0.00001)
    svhnModel.summary()

    callback = []
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/designedBGRClassifier.hdf5',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)
    tb = keras.callbacks.TensorBoard(log_dir = 'logs',
                                      write_graph = True,
                                      batch_size = batch_size,
                                      write_images = True)
    es = keras.callbacks.EarlyStopping(monitor= 'loss',  #'dig1_loss',
                                       min_delta=0.000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(tb)
    callback.append(es)
    callback.append(checkpointer)
    callback.append(reduce_lr)


    # svhnModel.fit_generator(
    #                   datagen.flow(ctrain, ctrlab, batch_size=batch_size),
    #                   batch_size = batch_size,
    #                   epochs=epochs,
    #                   verbose=1,
    #                   shuffle = True,
    #                   validation_data=(cvald, cvlab),
    #                   callbacks= callback)


    # fits the model on batches with real-time data augmentation:
    # svhnModel.fit_generator(datagen.flow(ctrain, ctrlab, batch_size=batch_size),
    #                         steps_per_epoch=len(ctrain) / batch_size,
    #                         epochs=epochs,
    #                         verbose=1,
    #                         validation_data = (cvald, cvlab),
    #                         callbacks= callback)
    #
    designHist = svhnModel.fit(x = trainX,
                              y = trainY,
                              batch_size = batch_size,
                              epochs = epochs,
                              verbose=1,
                              shuffle = True,
                              validation_data = (valdX, valdY),
                              callbacks= callback)

    print(designHist.history.keys())
    modName = 'customDesign'
    print(designHist.history.keys())
    createSaveMetricsPlot(designHist,modName,data,svhnModel)


def digitDetectorCNN():
    x, y = helper.preprocessDigDetector()
    _,row, col = x.shape
    channel = 1

    train = np.reshape(x,(x.shape[0],row,col,channel))
    numtrain = train.shape[0]

    epochs = 100
    batch_size = 64
    # optim = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    optim = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    # optim = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

    seed = 25
    np.random.seed(seed)
    split = np.int64(np.round((.95 * numtrain)))
    idx = np.random.permutation(numtrain-1)
    trIdx = idx[0:split]
    vlIdx = idx[split:numtrain]

    y = y.astype(dtype= 'int8')
    ctrain = train[trIdx]
    ctest  = train[vlIdx]

    # y = ku.to_categorical(y, 2)
    yTr = y[trIdx]
    yTs = y[vlIdx]

    datagen = ImageDataGenerator(
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range = 30,
                width_shift_range=0.5,
                height_shift_range=0.5,
                horizontal_flip=True,
                vertical_flip= True)
    datagen.fit(ctrain)

    input = keras.Input(shape=(row,col,channel), name='in')
    M = Conv2D(16,(3,3),activation='relu',padding='same')(input)
    M = Conv2D(16,(3, 3), activation ='relu', padding='same',name = 'conv1.5_128')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)
    M = Dropout(0.25)(M)

    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2_16')(M)
    M = Conv2D(32, (3, 3), activation ='relu', padding='same', name = 'conv2.5_16')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2))(M)
    M = Dropout(0.5)(M)

    M = Conv2D(64, (3, 3), activation ='relu', padding='same',name = 'conv2_32')(M)
    M = Conv2D(64, (3, 3), activation ='relu', padding='same', name = 'conv2.5_32')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)
    M = Dropout(0.25)(M)

    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv41_256')(M)
    M = Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same',name = 'conv4_256')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D(pool_size=(2, 2),strides = 1)(M)
    M = Dropout(0.5)(M)

    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'some256')(M)
    M = Conv2D(256, (5, 5), activation='relu', padding='same',name = 'some1256')(M)
    M = BatchNormalization(axis=-1)(M)
    M = MaxPooling2D((2, 2), strides= 1)(M)
    M = Dropout(0.25)(M)
    M = Flatten()(M)
    M = Dense(1024, activation='relu', name = 'FC1_1024')(M)
    M = Dense(512, activation='relu', name = 'FC2_1024')(M)
    M = Dropout(0.5)(M)

    out = Dense(1, activation='sigmoid',name = 'num')(M)


    digModel = keras.Model(inputs = input, outputs = out)

    digModel.compile(loss = 'binary_crossentropy',
                      optimizer= 'adam',
                      metrics=  ['accuracy'])
    digModel.summary()

    callback = []
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/weights.DigitsClassifier.hdf5',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)
    callback.append(checkpointer)

    # digModel.fit_generator(
    #                   datagen.flow(x = ctrain, y = yTr,batch_size = 32),
    #                   epochs=20,
    #                   verbose=1,
    #                   shuffle = True,
    #                   validation_data = (ctest, yTs),
    #                   callbacks= callback)

    # # fits the model on batches with real-time data augmentation:
    history = digModel.fit_generator(datagen.flow(ctrain, yTr, batch_size= 64),
                                     steps_per_epoch=len(ctrain) / 64,
                                     epochs=epochs,
                                     verbose = 1,
                                     validation_data = (ctest, yTs),
                                     callbacks= callback)
    #
    # digModel.fit(x = ctrain, y = yTr,
    #               batch_size = batch_size,
    #               epochs=epochs,
    #               verbose=1,
    #               shuffle = True,
    #               validation_data = (ctest, yTs),
    #               callbacks= callback)

    yOut = history.predict(ctrain)
    score = history.evaluate(ctrain, yTr, verbose=0)
    print(history.history.history.keys())


def ceLoss(y_true,y_predict):
     loss = K.mean(K.sparse_categorical_crossentropy(y_true,y_predict),axis=0)
     return loss


def predictImageNum(im,yOut,num):
    ply.imshow(im.squeeze())
    print([np.argmax(yOut[0][num]),
           np.argmax(yOut[1][num]),
           np.argmax(yOut[2][num]),
           np.argmax(yOut[3][num]),
           np.argmax(yOut[4][num])])


def new_accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1]
            / predictions.shape[0])


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def scratchVGG16_Model():
    data = helper.prepDataforCNN(numChannel = 3,feat_norm=True)
    trainX = data["trainX"]
    valdX  = data["valdX"]
    trainY = data["trainY"]
    valdY  = data["valdY"]

    _,row, col,channel = trainX.shape
    digLen = 5 # including category 0
    numDigits = 11
    epochs = 50
    batch_size = 64

    vgg16Model = VGG16(include_top = False,
                       weights = None)
    vgg16Model.summary()
    ptInput = keras.Input(shape = (row,col,channel), name  = 'vgg16Scratch')
    vgg16 = vgg16Model(ptInput)

    # vgg16 = Conv2D(64,(3, 3), activation ='relu', padding='same')(input)
    # vgg16 = Conv2D(64,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = MaxPooling2D(pool_size=(2, 2))(vgg16)
    #
    # vgg16 = Conv2D(128,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = Conv2D(128,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = MaxPooling2D(pool_size=(2, 2))(vgg16)
    #
    # vgg16 = Conv2D(256,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = Conv2D(256,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = MaxPooling2D(pool_size=(2, 2))(vgg16)
    #
    # vgg16 = Conv2D(512,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = Conv2D(512,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = Conv2D(512,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = MaxPooling2D(pool_size=(2, 2))(vgg16)
    #
    # vgg16 = Conv2D(512,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = Conv2D(512,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = Conv2D(512,(3, 3), activation ='relu', padding='same')(vgg16)
    # vgg16 = MaxPooling2D(pool_size=(2, 2))(vgg16)

    vgg16 = Flatten()(vgg16)
    vgg16 = Dense(512, activation='relu')(vgg16)
    vgg16 = Dense(512, activation='relu')(vgg16)
    # vgg16 = Dense(1000, activation='relu')(vgg16)
    vgg16 = Dropout(0.5)(vgg16)

    numd_SM = Dense(digLen,    activation='softmax',name = 'num')(vgg16)
    dig1_SM = Dense(numDigits, activation='softmax',name = 'dig1')(vgg16)
    dig2_SM = Dense(numDigits, activation='softmax',name = 'dig2')(vgg16)
    dig3_SM = Dense(numDigits, activation='softmax',name = 'dig3')(vgg16)
    dig4_SM = Dense(numDigits, activation='softmax',name = 'dig4')(vgg16)
    numB_SM = Dense(2,         activation='softmax',name = 'nC')(vgg16)
    out = [numd_SM, dig1_SM ,dig2_SM, dig3_SM, dig4_SM, numB_SM]

    vgg16 = keras.Model(inputs = ptInput, outputs = out)

    callback = []
    optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)

    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/vgg16.classifier.hdf5',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss',
                                                  factor = 0.1,
                                                  verbose = 1,
                                                  patience= 3,
                                                  cooldown= 0,
                                                  min_lr = 0.000001)
    # tb = keras.callbacks.TensorBoard(log_dir='logs', write_graph=True, write_images=True)
    es = keras.callbacks.EarlyStopping(monitor= 'val_loss',
                                       min_delta=0.00000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(es)
    callback.append(checkpointer)
    callback.append(reduce_lr)
    vgg16.summary()

    vgg16.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer= optim,
                  metrics=  ['accuracy'])

    vgg16History = vgg16.fit(x = trainX,
                             y = trainY,
                             batch_size = batch_size,
                             epochs=epochs,
                             verbose=1,
                             shuffle = True,
                             validation_data = (valdX, valdY),
                             callbacks = callback)

    print(vgg16History.history.keys())
    modName = 'vgg16_Scratch'
    print(vgg16History.history.keys())
    createSaveMetricsPlot(vgg16History,modName,data,vgg16)



def preTrainedVGG16_Model():
    data = helper.prepDataforCNN(numChannel = 3, feat_norm= True)
    trainX = data["trainX"]
    valdX  = data["valdX"]
    trainY = data["trainY"]
    valdY  = data["valdY"]

    _,row, col,channel = trainX.shape
    digLen = 5
    numDigits = 11
    epochs = 50
    batch_size = 64
    optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)


    preTrainModel = VGG16(include_top = False, weights = 'imagenet')
    preTrainModel.summary()
    ptInput = keras.Input(shape = (row,col,channel), name  = 'inputVGGPreTrain')
    pt_vgg16 = preTrainModel(ptInput)

    Mout = Flatten(name = 'flatten')(pt_vgg16)
    Mout = Dense(1024, activation='relu', name = 'FC1_4096')(Mout)
    Mout = Dense(1024, activation='relu', name = 'FC1_512')(Mout)
    # Mout = Dense(512,  activation='relu', name = 'FC2_1024')(Mout)
    # Mout = Dropout(0.5)(Mout)

    numd_SM = Dense(digLen,    activation='softmax',name = 'num')(Mout)
    dig1_SM = Dense(numDigits, activation='softmax',name = 'dig1')(Mout)
    dig2_SM = Dense(numDigits, activation='softmax',name = 'dig2')(Mout)
    dig3_SM = Dense(numDigits, activation='softmax',name = 'dig3')(Mout)
    dig4_SM = Dense(numDigits, activation='softmax',name = 'dig4')(Mout)
    numB_SM = Dense(2,         activation='softmax',name = 'nC')(Mout)
    out = [numd_SM, dig1_SM ,dig2_SM, dig3_SM, dig4_SM,numB_SM] #numd_SM

    vggPreTrain = keras.Model(inputs = ptInput, outputs = out)

    vggPreTrain.compile(loss = 'sparse_categorical_crossentropy', #ceLoss ,
                        optimizer= optim,
                        metrics=  ['accuracy']) #[])
    vggPreTrain.summary()

    callback = []
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='saved_models/VGGPreTrained.classifier.hdf5',
                                                   monitor='loss',
                                                   save_best_only=True,
                                                   verbose=2)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss',
                                                  factor = 0.1,
                                                  verbose = 1,
                                                  patience= 4,
                                                  cooldown= 1,
                                                  min_lr = 0.0001)
    es = keras.callbacks.EarlyStopping(monitor= 'loss',
                                       min_delta=0.000001,
                                       patience=5,
                                       verbose=1,
                                       mode='auto')
    callback.append(es)
    callback.append(checkpointer)
    callback.append(reduce_lr)

    vggHistory = vggPreTrain.fit(x = trainX,
                                 y = trainY,
                                 batch_size = batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 shuffle = True,
                                 validation_data = (valdX, valdY),
                                 callbacks= callback)

    print(vggHistory.history.keys())
    modName = 'vgg16_PreTrain'
    # list all data in history
    print(vggHistory.history.keys())
    createSaveMetricsPlot(vggHistory,modName,data,vggPreTrain)


def measurePrediction(out,label):
    labs = np.asarray(label).squeeze()
    numfeat, numsamp = labs.shape
    preds = []
    outY  = []
    for i in range(0,numfeat,1):
        val = np.argmax(out[i],axis=1).astype('uint8')
        preds.append(np.count_nonzero(val == labs[i].flatten())/numsamp * 100)
        outY.append(val)

    outYarr = np.asarray(outY).T
    seqAcc = np.count_nonzero(np.all(outYarr[:,1:5]==labs[1:5,:].T,axis=1))/ np.float(numsamp) * 100
    return preds, outY, seqAcc


def createSaveMetricsPlot(modelH,modName,data,model):
    trainX = data["trainX"]
    testX  = data["testX"]
    valdX  = data["valdX"]
    trainY = data["trainY"]
    testY  = data["testY"]
    valdY  = data["valdY"]

    ply.show()
    fig1 = ply.gcf()
    ply.ylim([0,1])
    ply.plot(modelH.history['dig1_acc'])
    ply.plot(modelH.history['val_dig1_acc'])
    ply.title('Digit1 accuracy')
    ply.ylabel('accuracy')
    ply.xlabel('epoch')
    ply.legend(['train', 'val'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelDig1Accuracy_'+ modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.ylim([0,1])
    ply.plot(modelH.history['dig2_acc'])
    ply.plot(modelH.history['val_dig2_acc'])
    ply.title('Digit2 accuracy')
    ply.ylabel('accuracy')
    ply.xlabel('epoch')
    ply.legend(['train', 'val'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelDig2Accuracy_'+ modName + '.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.ylim([0,1])
    ply.plot(modelH.history['dig3_acc'])
    ply.plot(modelH.history['val_dig3_acc'])
    ply.title('Digit3 accuracy')
    ply.ylabel('accuracy')
    ply.xlabel('epoch')
    ply.legend(['train', 'val'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelDig3Accuracy_'+ modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.ylim([0,1])
    ply.plot(modelH.history['dig4_acc'])
    ply.plot(modelH.history['val_dig4_acc'])
    ply.title('Digit4 accuracy')
    ply.ylabel('accuracy')
    ply.xlabel('epoch')
    ply.legend(['train', 'val'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelDig4Accuracy_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.ylim([0,1])
    ply.plot(modelH.history['num_acc'])
    ply.plot(modelH.history['val_num_acc'])
    ply.title('model accuracy')
    ply.ylabel('Number Digits Accuracy')
    ply.xlabel('epoch')
    ply.legend(['train', 'val'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelNumDigitsAccuracy_' + modName + '.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.plot(modelH.history['loss'])
    ply.plot(modelH.history['val_loss'])
    ply.title('Model loss')
    ply.ylabel('loss')
    ply.xlabel('epoch')
    ply.legend(['train', 'validation'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelLoss_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.plot(modelH.history['dig1_loss'])
    ply.plot(modelH.history["val_dig1_loss"])
    ply.title('Dig1 loss')
    ply.ylabel('loss')
    ply.xlabel('epoch')
    ply.legend(['train', 'validation'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/digit1Loss_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.plot(modelH.history['dig2_loss'])
    ply.plot(modelH.history["val_dig2_loss"])
    ply.title('Dig2 loss')
    ply.ylabel('loss')
    ply.xlabel('epoch')
    ply.legend(['train', 'validation'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/digit2Loss_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.plot(modelH.history['dig3_loss'])
    ply.plot(modelH.history["val_dig3_loss"])
    ply.title('Dig3 loss')
    ply.ylabel('loss')
    ply.xlabel('epoch')
    ply.legend(['train', 'validation'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/digit3Loss_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    ply.show()
    fig1 = ply.gcf()
    ply.plot(modelH.history['dig4_loss'])
    ply.plot(modelH.history["val_dig4_loss"])
    ply.title('Dig4 loss')
    ply.ylabel('loss')
    ply.xlabel('epoch')
    ply.legend(['train', 'validation'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/digit4Loss_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    # summarize history for loss
    ply.show()
    fig1 = ply.gcf()
    ply.ylim([0,1])
    ply.plot(modelH.history['nC_acc'])
    ply.plot(modelH.history['val_nC_acc'])
    ply.title('Digit Classifier accuracy')
    ply.ylabel('accuracy')
    ply.xlabel('epoch')
    ply.legend(['train', 'val'], loc='upper left')
    ply.draw()
    fig1.savefig('plots/modelDigitClassifierAccuracy_' + modName +'.png', bbox_inches='tight', dpi= 200)
    ply.close()

    yOutr = model.predict(trainX)
    scoreTr = model.evaluate(trainX, trainY, verbose=0)
    trainpAcc, outYt, seqTrainAcc = measurePrediction(yOutr,trainY)
    print('Train loss:', scoreTr[0])
    print('numdigits', 'digit1','digit2','digit3','digit4')
    print('Train accuracy:', trainpAcc)
    print('Train sequence accuracy:' , seqTrainAcc)

    yOuts = model.predict(testX)
    testAcc, outYtest, seqTestPred = measurePrediction(yOuts,testY)
    scoreTest = model.evaluate(testX, testY, verbose=0)
    print('Test loss:', scoreTest[0])
    print('numdigits', 'digit1','digit2','digit3','digit4')
    print('Test per digit accuracy:', testAcc)
    print('Test sequence accuracy:' , seqTestPred)

    yOutv = model.predict(valdX)
    valAcc, outYtest, seqValPred = measurePrediction(yOutv,valdY)
    scoreV = model.evaluate(valdX, valdY, verbose=0)
    print('Validation loss:', scoreV[0])
    print('numdigits', 'digit1','digit2','digit3','digit4')
    print('Validation per digit accuracy:', valAcc)
    print('VAlidation sequence accuracy:' , seqValPred)

    metrics = {'trainAcc'   : trainpAcc,
               'testAcc'    : testAcc,
               'valAcc'     : valAcc,
               'trainSeqAcc': seqTrainAcc,
               'testSeqAcc' : seqTestPred,
               'valSeqAcc'  : seqValPred,
               'trainScore' : scoreTr,
               'testScore'  : scoreTest,
               'valScore'   : scoreV}
    # np.save('metrics/' + modName +'.npy', metrics)
    with open('metrics/' + modName +'.pickle', 'wb') as handle:
         pickle.dump(metrics, handle, protocol = pickle.HIGHEST_PROTOCOL)

    with open('metrics/' + modName +'History.pickle', 'wb') as handle:
        pickle.dump(modelH.history, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
     # run_Detection()
     # setupCNNforDigits()
     # setupCNNforSequenceLength()
     # helper.resizeSamples()
     # digitDetectorCNN()
     # scratchVGG16_Model()
     designedCNN_Model()
     #detection.runSVHNDetection(13)
     # detection.loadAndDetectImages()
     # preTrainedVGG16_Model()
     # detection.createCNNVideo()
