import math
import operator


def calcShannonEnt(dataset):
    numEnt = len(dataset)   # 返回数据集行数
    labelCounts = {}        # 保存每个标签出现的次数
    for featVec in dataset: # 对每组特征向量进行统计
        currLabel = featVec[-1]
        if currLabel not in labelCounts.keys(): # 如果标签没有放入统计次数的字典里，添加进去
            labelCounts[currLabel] = 0
        labelCounts[currLabel] += 1
    shannonEnt = 0.0        # 经验熵（香浓熵）
    for key in labelCounts: # 计算香浓熵
        prob = float(labelCounts[key]) / numEnt # 5/13 = 0.36   8/13 = 0.618
        # 该标签的概率
        shannonEnt -= prob * math.log(prob, 2)       # 公式计算
    return shannonEnt


def createDataSet():
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 1, 1, 0, 'yes'],
               [1, 0, 1, 0, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']
               ]
    labels = ['年龄', '有工作', '有自己房子', '信贷情况']
    return dataset, labels


def splitDataSet(dataset, axis, value):
    retDataSet=[]   # 创建返回的数据集列表
    for featVec in dataset:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]              # 去掉axis标签
            reduceFeatVec.extend(featVec[axis + 1:])    # 将符合条件的添加到返回的数据集
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1       # 特征数
    baseEntropy = calcShannonEnt(dataset)   # 计算数据集的香浓熵
    bestInfoGain = 0.0                      # 信息增益
    bestFeature = -1                        # 最优特征的索引值
    for i in range(numFeatures):
        # 获取dataset第i个所有特征--第i列全部特征
        featList = [example[i] for example in dataset]
        uniqueVals = set(featList)          # 创建集合，元素不重复
        newEntropy = 0.0                    # 条件熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataset, i, value)    # subDataset划分后的子集
            prob = len(subDataSet) / float(len(dataset))    # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 计算条件熵
        infoGain = baseEntropy - newEntropy                 # 信息增益

        if(infoGain > bestInfoGain):        # 计算信息增益
            bestInfoGain = infoGain         # 更新信息增益，找到最大的信息增益
            bestFeature = i                 # 记录信息增益最大的特征的索引
    return bestFeature



def majorityCnt(classList):
    # 当特征划分到只有一个，但无法归为一类时，即无法停止构造，采取最大投票法，选择最多类别作为该类标签
    classCount = {}
    for vote in classList:              # 统计classList中每个元素出现的次数
        if vote in classList.keys():
            classList[vote] = 0
        classList[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataset, labels, featLabels):
    classList = [example[-1] for example in dataset]
    if classList.count(classList[0]) == len(classList): # 如果类别完全相同则停止划分
        return classList[0]
    if len(dataset[0]) == 1:                            # 遍历完所有特征时返回出现次数最多的特征
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataset)        # 选择最优特征
    bestFeatLabel = labels[bestFeat]                    # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}                        # 根据最优特征生成树
    del (labels[bestFeat])                              # 删除已使用的标签
    featValues = [example[bestFeat] for example in dataset] # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)                        # 去掉重复的属性值
    for value in uniqueVals:                            # 遍历特征，创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataset, bestFeat, value), labels, featLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    dataset, labels = createDataSet()

    print('最优特征索引值：' + str(chooseBestFeatureToSplit(dataset)))

    featLabels = []
    myTree = createTree(dataset, labels, featLabels)

    print(myTree)

    testVec = [0, 0, 1, 1]
    res = classify(myTree, featLabels, testVec)
    if res == 'yes':
        print('fang')
    if res == 'no':
        print('no')
