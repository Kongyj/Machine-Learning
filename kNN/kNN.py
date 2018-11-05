import numpy as np
import matplotlib.pyplot as plt
import os


#read training data

def file2matrix(filename):
    fr = open(filename)
    linesize = len(fr.readlines())
    returnmatrix = np.zeros((linesize, 3))
    fr.close()
    fr = open(filename)
    index = 0
    returnlabel = []
    for line in fr.readlines():

        line = line.strip()
        line = line.split('\t')
        returnmatrix[index, :] = list(map(float, line[:-1]))
        index += 1
        #results = list(map(int, results))
        if line[-1] == 'largeDoses':
            returnlabel.append(3)
        elif line[-1] == 'smallDoses':
            returnlabel.append(2)
        else:
            returnlabel.append(1)
    fr.close()
    return returnmatrix, returnlabel


def viewdata(datamat, datalabel):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Datedata Plot')
    size1 = len(datamat)
    type1, type2, type3 = [], [], []
    label1, label2, label3 = [], [], []
    for i in range(size1):
        if datalabel[i] == 1:
            type1.append(list(datamat[i]))
            label1.append(datalabel[i])
        elif datalabel[i] == 2:
            type2.append(list(datamat[i]))
            label2.append(datalabel[i])
        else:
            type3.append(list(datamat[i]))
            label3.append(datalabel[i])
    type1, type2, type3 = np.array(type1),np.array(type2),np.array(type3)
    type_1 = ax.scatter(type1[:, 1], type1[:, 2], s = 20, c = 'red')
    type_2 = ax.scatter(type2[:, 1], type2[:, 2], s =40, c='green')
    type_3 = ax.scatter(type3[:, 1], type3[:, 2], s =60, c='blue')
    ax.legend([type_1, type_2, type_3], ["Did Not Like", "Liked in Small Doses", "Liked in Large Doses"], loc=2)
    #ax.scatter(datamat[:,1], datamat[:,2], 15.0*np.array(datalabel), 15.0*np.array(datalabel))
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    plt.savefig("dateplot.jpg")
    plt.show()

def autonorm(datamat):
    maxinmun = datamat.max(axis = 0)
    minimun = datamat.min(axis = 0)
    datanorm = (datamat - minimun)/(maxinmun - minimun)
    return datanorm, maxinmun, minimun

TESTNUM = 200
def file2array(path):
    #read in all file in traindata and transfer to array
    path = path
    files = os.listdir(path)
    returnlabel = []
    returnmat = []
    index = 0
    testnum = 0
    for file in files:

        if not os.path.isdir(file):
            label = int(file[0])
            returnlabel.append(label)
            f = open(path+'/'+file)
            row = 0
            for line in f.readlines():
                if row < 32:
                    line = line.strip('\n')
                    for i in line:
                        returnmat.append(i)
                    row += 1
            index += 1
    returnmat = list(map(int, returnmat))
    returnmat = np.array(returnmat)
    returnmat.resize(index, 32*32)
    return returnmat, returnlabel

def datingcls():
    filename = "datingTestSet.txt"
    datamat, datalable = file2matrix(filename)
    viewdata(datamat, datalable)
    datamat = np.array(datamat)
    datanorm, maxinmun, minimun = autonorm(datamat)
    testdata = [14488, 7.153469, 1.673904]
    testnorm = (testdata - minimun) / (maxinmun - minimun)
    datalable = np.array(datalable)
    datalable = datalable - 1
    result_label = KNN(testnorm, datanorm, datalable, 3, 7)
    label = ['did no like', 'small like', 'large like']
    print('how much you like him:', label[result_label])

def KNN(indx, datanorm, datalabel, label_num, k):
    #compute dist
    datasize = len(datanorm)
    dist = np.tile(indx, datasize).reshape(datasize, -1) - datanorm
    sqdist = np.sum(dist**2, axis=1)
    dist = np.sqrt(sqdist)
    distidx = dist.argsort()
    classCount = np.zeros(label_num)
    for i in range(k):
        label = datalabel[distidx[i]]
        classCount[label] = classCount[label] + 1
    classCount = np.array(classCount.astype(int))
    resultlabel = np.argsort(-classCount)[0]
    return resultlabel

def test(path, datamat, datalabel):
    files = os.listdir(path)
    correctness = 0
    testnum = 0
    datasize = len(datalabel)
    for file in files:
        if testnum <TESTNUM:
            testnum += 1
            if not os.path.isdir(file):
                truelabel = int(file[0])
                fr = open(path + '/' + file)
                subtest = []
                row = 0
                for line in fr.readlines():
                    if row < 32:
                        line = line.strip('\n')
                        for i in line:
                            subtest.append(i)
                        row += 1
                subtest = list(map(int, subtest))
                subtest = np.array(subtest)

                #subtest = subtest.resize(1, 32*32)
                #subtest = list(subtest)
                prelabel = KNN(subtest, datamat, datalabel, 10, 7)
                #print(prelabel, truelabel)
                if prelabel == truelabel:
                    correctness += 1
                fr.close()
    correct = correctness / TESTNUM
    print('the pecentage of correctness of recognize minist:', correct)

def writetest():
    path = './trainingDigits'
    datamat, datalabel = file2array(path)
    path = './testDigits'
    test(path, datamat, datalabel)

#view the plot of data
if __name__=='__main__':
    #filename = input("please input the filename: ")
    datingcls()
    writetest()




