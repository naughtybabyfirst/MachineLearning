import numpy as np
import time

def create_data():
    group = np.array([[110, 4], [119, 6], [10, 100], [9, 110]])
    label = ['爱', '爱', '动', '动']
    return group, label


def classify(testD, trainD, trainL, k):
    '''
    :param testD: 测试集数据
    :param trainD: 训练集数据
    :param trainL: 训练集标签
    :param k:
    :return:
    '''
    trainD_Size = trainD.shape[0]
    # 求差值
    diffMat = np.tile(testD, (trainD_Size, 1)) - trainD
    # 平方
    squDiff = diffMat ** 2
    # 求和
    squSum = squDiff.sum(axis=1)
    # 开方
    dist = squSum ** 0.5
    # 对dist排序
    sortedDist = dist.argsort()

    classCount = {}
    for i in range(k):
        votelabel = label[sortedDist[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1

    sortedclasscount = sorted(classCount.items(), reverse=True)

    return sortedclasscount[0][0]


if __name__ == '__main__':
    start_time = time.time()
    group, label = create_data()
    test = [90, 3]
    testRes = classify(test, group, label, 3)
    print('time:', time.time() - start_time)
    print(testRes)
