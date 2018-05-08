import cv2
import numpy as np
import scipy.io as sc
import os
import h5py
import pickle
import detection
import matplotlib.pyplot as plt


def extract_trainRGB():
    train = os.path.join("train")
    dirname = 'finalProjectData/train/'
    out = sc.loadmat(os.path.join(dirname,'trainDigits.mat'))
    data = out["digitStruct"]
    bbox = data['bbox'].squeeze()
    name = data['name'].squeeze()
    _,dm = data.shape

    negimg = []
    tr_label = []
    img = []
    tr_labels5 = []
    imgBW    = []
    negimgBW = []

    sz = (48,48)
    neglabel = np.asarray([0, 10, 10, 10, 10, 10],dtype = 'uint8')
    filler = [0, 10, 10, 10, 10, 10]
    # sz2 = (224,224)

    hf = h5py.File('datasets/train.h5', 'w')

    for i in range(0,dm,1):  #dm
        im = cv2.imread(os.path.join(dirname,name[i][0].squeeze()))
        b = bbox[i].flatten()
        ht,wd,_ = im.shape

        ha = np.max([0, np.int16(np.min(b["top"].squeeze()).squeeze())])
        wa = np.max([0, np.int16(np.min(b["left"].squeeze()).squeeze())])
        hb = np.int16(np.max(b["height"].squeeze()).squeeze()) + ha
        wb = np.int16(np.sum(b["width"].squeeze()).squeeze())  + wa
        h3 = (hb - ha) * .3
        w3 = (wb - wa) * .3
        ha = np.max([0,  np.int16(ha - h3)])
        hb = np.min([ht, np.int16(hb + h3)])
        wa = np.max([0,  np.int16(wa - w3)])
        wb = np.min([wd, np.int16(wb + w3)])
        if (wb-wa == 0) | (hb-ha == 0):
           continue

        numdig = b["label"].__len__()
        if numdig>=5:
           continue  # skip >5 digits

        bboxIm = im[ha:hb,wa:wb,:]
        totIm = cv2.resize(bboxIm,sz)
        img.append(totIm)
        imgBW.append(cv2.cvtColor(totIm,cv2.COLOR_BGR2GRAY))

        lab = np.copy(filler)
        lab[0] = b["label"].__len__()

        for n in range(0,numdig,1):
            currlab = np.int16(b["label"][n].squeeze())
            if currlab == 10:
               currlab = 0
            lab[n+1] = currlab

        currLabel = np.asarray(lab, dtype='uint8')
        tr_labels5.append(currLabel)


        if (wa > 10) & (ha >10):
           cropIm = cv2.resize(im[0: ha-1, 0:wa-1 ,:],sz)
           negimg.append(cropIm)
           tr_label.append(neglabel)
           negimgBW.append(cv2.cvtColor(cropIm, cv2.COLOR_BGR2GRAY))

           cropIm = cv2.resize(im[ha:hb, 0:wa-1,:], sz)
           negimg.append(cropIm)
           negimgBW.append(cv2.cvtColor(cropIm, cv2.COLOR_BGR2GRAY))
           tr_label.append(neglabel)

           cropIm = cv2.resize(im[0:ha-1, wa:wb,:], sz)
           negimg.append(cropIm)
           negimgBW.append(cv2.cvtColor(cropIm, cv2.COLOR_BGR2GRAY))
           tr_label.append(neglabel)

        if (ht-hb > 10) & ((wd -wb)>10):
           cropIm = cv2.resize(im[hb+1: ht, wb+1:wd,:],sz)
           negimg.append(cropIm)
           negimgBW.append(cv2.cvtColor(cropIm, cv2.COLOR_BGR2GRAY))
           tr_label.append(neglabel)

           cropIm = cv2.resize(im[ha: hb, wb+1:wd,:],sz)
           negimg.append(cropIm)
           negimgBW.append(cv2.cvtColor(cropIm, cv2.COLOR_BGR2GRAY))
           tr_label.append(neglabel)

           cropIm = cv2.resize(im[hb+1: ht, wa:wb,:],sz)
           negimg.append(cropIm)
           negimgBW.append(cv2.cvtColor(cropIm, cv2.COLOR_BGR2GRAY))
           tr_label.append(neglabel)

        print(i)

    hf.create_dataset('negdigits',   data=negimg)
    hf.create_dataset('digits',      data=img)
    hf.create_dataset('negdigitsBW', data=negimgBW)
    hf.create_dataset('digitsBW',    data=imgBW)
    hf.create_dataset('neglab',      data=tr_label)
    hf.create_dataset('labs5',       data=tr_labels5)
    hf.close()

def extract_testRGB():
    test = os.path.join("test")
    # dirname = 'finalProjectData/test/'
    # out = sc.loadmat('finalProjectData/test/testDigits.mat')
    dirname = 'test/'
    out = sc.loadmat('test/testDigits.mat')
    data = out["digitStruct"]
    bbox = data['bbox'].squeeze()
    name = data['name'].squeeze()
    _,dm = data.shape

    img = []
    tr_labels5 = []
    imgBW    = []

    sz = (48,48)
    filler = [0, 10, 10, 10, 10, 10]
    hf = h5py.File('datasets/test.h5', 'w')

    for i in range(0,dm,1):  #dm
        im = cv2.imread(os.path.join(dirname,name[i][0].squeeze()))
        b = bbox[i].flatten()
        ht,wd,_ = im.shape

        ha = np.max([0, np.int16(np.min(b["top"].squeeze()).squeeze())])
        wa = np.max([0, np.int16(np.min(b["left"].squeeze()).squeeze())])
        hb = np.int16(np.max(b["height"].squeeze()).squeeze()) + ha
        wb = np.int16(np.sum(b["width"].squeeze()).squeeze())  + wa
        h3 = (hb - ha) * .3
        w3 = (wb - wa) * .3
        ha = np.max([0,  np.int16(ha - h3)])
        hb = np.min([ht, np.int16(hb + h3)])
        wa = np.max([0,  np.int16(wa - w3)])
        wb = np.min([wd, np.int16(wb + w3)])
        if (wb-wa == 0) | (hb-ha == 0):
            continue

        numdig = b["label"].__len__()
        if numdig>=5:
           continue  # skip >5 digits

        bboxIm = im[ha:hb,wa:wb,:]
        totIm = cv2.resize(bboxIm,sz)
        img.append(totIm)
        imgBW.append(cv2.cvtColor(totIm,cv2.COLOR_BGR2GRAY))

        lab = np.copy(filler)
        lab[0] = b["label"].__len__()

        for n in range(0,numdig,1):
            currlab = np.int16(b["label"][n].squeeze())
            if currlab == 10:
               currlab = 0
            lab[n+1] = currlab

        currLabel = np.asarray(lab,dtype = 'uint8')
        tr_labels5.append(currLabel)

        print(i)

    hf.create_dataset('digits', data=img)
    hf.create_dataset('digitsBW', data=imgBW)
    hf.create_dataset('labs5', data=tr_labels5)
    hf.close()


def extract_extraTrainRGB():
    # out = sc.loadmat('E:/extra/digitStruct.mat')
    # data = out["digitStruct"]
    out = sc.loadmat('digitStruct.mat')
    data = out["digitStruct"]
    extrain = "E:/extra"
    bbox = data['bbox'].squeeze()
    name = data['name'].squeeze()
    _, dm = data.shape

    img = []
    tr_labels5 = []
    imgBW    = []

    sz = (48,48)
    filler = [0, 10, 10, 10, 10, 10]

    # open the file for writing
    hf = h5py.File('datasets/extraTrain.h5', 'w')

    for i in range(0,dm,1):  #dm
        im = cv2.imread(os.path.join(extrain,name[i][0].squeeze()))
        b = bbox[i].flatten()
        ht,wd,_ = im.shape

        ha = np.max([0, np.int16(np.min(b["top"].squeeze()).squeeze())])
        wa = np.max([0, np.int16(np.min(b["left"].squeeze()).squeeze())])
        hb = np.int16(np.max(b["height"].squeeze()).squeeze()) + ha
        wb = np.int16(np.sum(b["width"].squeeze()).squeeze())  + wa
        h3 = (hb - ha) * .3
        w3 = (wb - wa) * .3
        ha = np.max([0,  np.int16(ha - h3)])
        hb = np.min([ht, np.int16(hb + h3)])
        wa = np.max([0,  np.int16(wa - w3)])
        wb = np.min([wd, np.int16(wb + w3)])
        if (wb-wa == 0) | (hb-ha == 0):
            continue

        numdig = b["label"].__len__()
        if numdig>=5:
            continue  # skip >5 digits

        bboxIm = im[ha:hb,wa:wb,:]
        totIm = cv2.resize(bboxIm,sz)
        img.append(totIm)
        imgBW.append(cv2.cvtColor(totIm,cv2.COLOR_BGR2GRAY))

        lab = np.copy(filler)
        lab[0] = b["label"].__len__()

        for n in range(0,numdig,1):
            currlab = np.int16(b["label"][n].squeeze())
            if currlab == 10:
                currlab = 0
            lab[n+1] = currlab

        currLabel = np.asarray(lab,dtype = 'uint8')
        tr_labels5.append(currLabel)

        print(i)

    hf.create_dataset('digits', data=img)
    hf.create_dataset('digitsBW', data=imgBW)
    hf.create_dataset('labs5', data=tr_labels5)
    hf.close()


def preprocessDigDetector():
    htr = h5py.File('datasets/training.h5', 'r')
    hts = h5py.File('datasets/testing.h5', 'r')
    htn = h5py.File('datasets/trainNegatives.h5', 'r')

    numNeg = range(0,35000,1)
    negLabs = htn["labs"][numNeg]
    x  = np.vstack((htr["train48"][:],hts["digit48"][:],htn["digit48"][numNeg])).astype('float32')
    y  = np.vstack((np.reshape(htr["labs5"][:,0],(len(htr["labs5"][:,0]),1)),
                    np.reshape(hts["labs5"][:,0],(len(hts["labs5"][:,0]),1)),
                    np.reshape(negLabs[:,0] ,(len(negLabs),1)))).astype('uint8')

    y[y>0] = 1
    # x = x/255.

    for i in range(x.shape[0]):
        x[i]-= np.mean(x[i].flatten(),axis = 0)

    x = x - np.mean(x,axis=0)
    x = x/np.std(x,axis = 0)

    return x,y

def prepDataforCNN(numChannel = 1, feat_norm = False):

    htr  = h5py.File('datasets/train.h5', 'r')
    hts  = h5py.File('datasets/test.h5' , 'r')
    hte =  h5py.File('datasets/extraTrain.h5', 'r')

    if numChannel == 1:
       digits     = htr["digitsBW"]
       testdigits = hts["digitsBW"]
       negdigits  = htr["negdigitsBW"]
       extdigits  = hte["digitsBW"]
    else:
       digits     = htr["digits"]
       testdigits = hts["digits"]
       negdigits  = htr["negdigits"]
       extdigits  = hte["digits"]

    trainlabs  = htr["labs5"]
    testlabs   = hts["labs5"]
    neglabs    = htr["neglab"]
    extlabs    = hte["labs5"]

    digits     = digits[:]
    testdigits = testdigits[:]
    negdigits  = negdigits[:]
    extdigits  = extdigits[:]
    trainlabs  = trainlabs[:]
    testlabs   = testlabs[:]
    neglabs    = neglabs[:]
    negdigits  = negdigits[:]

    seed = 25
    np.random.seed(seed)
    countNeg = 30000
    countX   = 90000

    negIdx = np.random.randint(0,negdigits.shape[0],countNeg)
    numNegTs = np.arange(negdigits.shape[0] - 500,negdigits.shape[0],1)
    numtran = np.arange(0,countX,1)  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) % Tried on 12K
    # numtest = np.arange(extdigits.shape[0] - 1,extdigits.shape[0],1)

    # negIdx = np.random.randint(0,negdigits.shape[0],3000)# Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) Tried on 12k
    # numtran = range(0,50,1)  # Ran on 0. Tried on 80000(with 2 512). % Tried on 30000(50%) % Tried on 12K

    xtrdigits = extdigits[numtran,:]
    xtrlab    = extlabs[numtran,:]

    ntrdigits = negdigits[negIdx]
    ntrlab    = neglabs[negIdx,:]
    ntest     = negdigits[numNegTs,:]
    ntslab    = neglabs[numNegTs,:]

    preTrain = np.vstack((digits,xtrdigits,ntrdigits)).astype('float32')  #
    preTest =  np.vstack((testdigits,ntest)).astype('float32')  #xtest

   # Lets remove digits > 4. Only 9 cases of n = 5
    trainlabs = np.vstack((trainlabs,xtrlab,ntrlab)).astype('uint8')  #xtrlab
    testlabs  = np.vstack((testlabs,ntslab)).astype('uint8')   #xtslab
    ind = np.argwhere(trainlabs[:,0]<5)
    ind = ind[:,0]
    preTrain = preTrain[ind,:]
    trainlabs = trainlabs[ind,:]

    nb = np.reshape(np.asarray(trainlabs[:,0] > 0, dtype = 'uint8'),(trainlabs.shape[0],1))
    trl = np.hstack((trainlabs, nb))

    ind = np.argwhere(testlabs[:,0]<5)
    ind = ind[:,0]
    preTest = preTest[ind,:]
    testlabs = testlabs[ind,:]

    nb = np.reshape(np.asarray(testlabs[:,0] > 0, dtype = 'uint8'),(testlabs.shape[0],1))
    tsl = np.hstack((testlabs, nb))

    # train = np.float64(preTrain/255.)
    # test  = np.float64(preTest/255.)
    train = np.float64(preTrain)
    test  = np.float64(preTest)

    for i in range(preTrain.shape[0]):
        if numChannel >1:
            for channel in range(0,numChannel,1):
                train[i][:,:,channel] -= np.mean(preTrain[i][:,:,channel].flatten(),axis = 0)
        else:
            train[i] -= np.mean(preTrain[i].flatten(),axis = 0)

    for i in range(preTest.shape[0]):
        if numChannel > 1:
           for channel in range(0,numChannel,1):
               test[i][:,:,channel] -= np.mean(preTest[i][:,:,channel] .flatten(),axis = 0)
        else:
           test[i] -= np.mean(preTest[i].flatten(),axis = 0)


    if feat_norm:
       M = np.mean(train, axis = 0)
       train = train - M
       sd = np.std(train, axis = 0)
       train = train/sd

       test = test - M
       test = test/sd
       featNorm = {'mean': M, 'std': sd}
       if numChannel > 1:
           with open('datasets/BGRnorm.pickle', 'wb') as handle:
                pickle.dump(featNorm, handle, protocol = pickle.HIGHEST_PROTOCOL)
       else:
           with open('datasets/BWnorm.pickle', 'wb') as handle:
                pickle.dump(featNorm, handle, protocol = pickle.HIGHEST_PROTOCOL)
                 # np.save('datasets/BWnorm.npz', featNorm)

    numtrain = train.shape[0]
    numtest  = test.shape[0]
    row = train.shape[1]
    col = train.shape[2]

    train = np.reshape(train,(numtrain,row,col,numChannel))
    test  = np.reshape(test, (numtest, row,col,numChannel))

    p = 0.9
    seed = 25
    np.random.seed(seed)
    split = np.int32(np.round((p * numtrain)))  #.85

    idx = np.random.permutation(numtrain)
    trIdx = idx[0:split]
    vlIdx = idx[split:numtrain]

    trlab = [ np.reshape(trl[:,0],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,1],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,2],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,3],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,4],(numtrain,1)).astype('uint8'),
              np.reshape(trl[:,6],(numtrain,1)).astype('uint8')]

    tslab = [ np.reshape(tsl[:,0],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,1],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,2],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,3],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,4],(numtest,1)).astype('uint8'),
              np.reshape(tsl[:,6],(numtest,1)).astype('uint8')]

    ctrlab = [trlab[0][trIdx], trlab[1][trIdx], trlab[2][trIdx], trlab[3][trIdx], trlab[4][trIdx], trlab[5][trIdx]]
    cvlab  = [trlab[0][vlIdx], trlab[1][vlIdx], trlab[2][vlIdx], trlab[3][vlIdx], trlab[4][vlIdx], trlab[5][vlIdx]]
    ctslab = [tslab[0],        tslab[1],        tslab[2],        tslab[3],        tslab[4],        tslab[5]]

    data = {'trainX': train[trIdx], 'trainY': ctrlab,
            'testX':  test,         'testY':  ctslab,
            'valdX':  train[vlIdx], 'valdY':  cvlab}

    return data
