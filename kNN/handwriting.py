import os
import numpy as np


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    with open(filename) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumberStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumberStr)
        trainingMat[i, :] = img2vector('traingDigits/%s' % fileNameStr)
