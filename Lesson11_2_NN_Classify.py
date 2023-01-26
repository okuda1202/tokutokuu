# Kobe University
# Advanced Course on Image Processing
# 2023.1.22

import cv2 as cv 
import numpy as np 

jpChar = [
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を", "ん"] 

#------------------------------------ 
nn = cv.ml.ANN_MLP_load("NetModel.xml")

# 46文字×50人分を準備
_46x50_chars = cv.imread("JPN46chars.bmp", cv.IMREAD_GRAYSCALE)

sample = np.zeros((1,400), 'float32')
labels = np.zeros((1,46), 'float32')

successNum, failNum = 0, 0

for y in range(0, 46):
    for x in range(0, 50):
        moji = _46x50_chars[y*20:y*20+20, x*20:x*20+20]
        cv.imshow("Win1", moji)
        
        for i in range(0, 400):         # 400画素を水平に並べかえる。
            sample[0,i] = moji[i//20, i%20]

        nn.predict(sample, labels)      # 「あ」～「ん」の確率を推定する。　

        min, max, minLoc, maxLoc = cv.minMaxLoc(labels) # 確率が最大となる位置（インデックス）を探す
        print(jpChar[maxLoc[0]], end=" ")               # それが答え

        if maxLoc[0] == y:
            successNum += 1
        else:
            failNum += 1

        cv.waitKey(1)

print("Success =", successNum, "Fail =", failNum, "認識率 =", successNum / (successNum + failNum)  )
cv.destroyAllWindows()    #なるべく呼ぶこと。
