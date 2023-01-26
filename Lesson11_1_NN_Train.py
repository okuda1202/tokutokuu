# Kobe University
# Advanced Course on Image Processing
# 2023.1.22

import cv2 as cv 
import numpy as np 

#------------------------------------ 
def MakeModel():
    #ニューラルネットワークを設計する
    nn = cv.ml.ANN_MLP_create()
    
    layer_size = np.zeros((3), 'int32')
    layer_size[0] = 400     #入力層
    layer_size[1] = 23      #中間層
    layer_size[2] = 46      #出力層

    nn.setLayerSizes(layer_size)
    nn.setActivationFunction(cv.ml.ANN_MLP_SIGMOID_SYM, 0.3, 1)

    return nn

#------------------------------------ 
def Load_trainData(fileName):
    # 46文字×50人分を教師データとして準備
    _46x50_chars = cv.imread(fileName, cv.IMREAD_GRAYSCALE)

    trainData = np.zeros((46*50, 400), 'float32')
    labels = np.zeros((46*50, 46), 'float32')

    count = 0

    for y in range(0, 46):              # あ:y=0, い:y=1, う:y=2, ...
        for x in range(0, 50):
            moji = _46x50_chars[y*20:y*20+20, x*20:x*20+20]     # 一文字ずつ切り取る

            cv.imshow("Win1", moji)
            labels[count, y] = 1.0       # あ:y=0, い:y=1, う:y=2, ...の位置に1を立てる。
            
            for i in range(0, 400):                     # 400画素を水平に並べかえる。
                trainData[count,i] = moji[i//20, i%20]
                
            count += 1
            cv.waitKey(1)

    return trainData, labels

#------------------------------------ 
def train(nn, trainData, labels):
    criteria = (cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 30, 0.001)
    nn.setTermCriteria(criteria)
    nn.setTrainMethod(cv.ml.ANN_MLP_BACKPROP, 0.1, 0.1)
    nn.train(trainData, cv.ml.ROW_SAMPLE, labels)

#------------------------------------ 
nn = MakeModel()
trainData, labels = Load_trainData("JPN46chars.bmp") 
train(nn, trainData, labels)
nn.save("NetModel.xml")   #学習済みファイルを保存
cv.destroyAllWindows()    #なるべく呼ぶこと。
