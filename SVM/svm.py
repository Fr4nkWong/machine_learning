import numpy as np
import random
import matplotlib.pyplot as plt


def load_data(dir):
    """
    加载数据集
    :param dir - 数据集的存放路径
    :return dataset - 处理好的样本列表
    """
    fr = open(dir)
    line_list = fr.readlines()
    labels = []
    dataset = []
    for line in line_list:
        line = line.strip()
        attr_list = line.split('\t')
        labels.append(int(attr_list[-1]))
        attr_list = [float(attr) for attr in attr_list[:-1]]
        dataset.append(attr_list)
    return dataset, labels


def select_jrand(i, m):
    """
    随机选择alpha
    :param i - alpha_i的索引值
    :param m - alpha参数个数
    :return j - alpha_j的索引值
    """
    j = i   #选择一个不等于i的j
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj,H,L):
    """
    裁剪alpha
    :param aj - alpha_j值
    :param H - alpha上限
    :param L - alpha下限
    :return aj - alpah值
    """
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj


def smo(dataset, labels, c, toler, max_iter):
    """
    SMO算法(简明版)
    :param dataset - 数据集
    :param labels - 标记列表
    :param c - 松弛变量
    :param toler - 容错率
    :param max_iter - 最大迭代次数
    :return b - 处理好的样本列表
    :return alphas - 
    """
    #转换为numpy的mat存储
    data_matrix = np.mat(dataset)   # m*n
    label_vec = np.mat(labels).transpose()  # m*1
    b = 0   #初始化b参数，统计数据矩阵的维度
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m,1)))    #初始化alpha参数，设为0 m*1
    iter_num = 0    #初始化迭代次数
    while (iter_num < max_iter):
        alphaPairsChanged = 0
        for i in range(m):
            #步骤1：计算误差Ei
            fXi = float(np.multiply(alphas,label_vec).T*(data_matrix*data_matrix[i,:].T)) + b
            Ei = fXi - float(label_vec[i])
            #优化alpha，设定一定的容错率
            if ((label_vec[i]*Ei < -toler) and (alphas[i] < c)) or ((label_vec[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选择另一个与alpha_i成对优化的alpha_j
                j = select_jrand(i,m)
                #步骤1：计算误差Ej
                fXj = float(np.multiply(alphas,label_vec).T*(data_matrix*data_matrix[j,:].T)) + b
                Ej = fXj - float(label_vec[j])
                #保存更新前的aplpha值，使用深拷贝
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy()
                #步骤2：计算上下界L和H
                if (label_vec[i] != label_vec[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = min(c, alphas[j] + alphas[i])
                if L==H: 
                    print("L==H")
                    continue
                #步骤3：计算eta
                eta = 2.0 * data_matrix[i,:]*data_matrix[j,:].T - data_matrix[i,:]*data_matrix[i,:].T - data_matrix[j,:]*data_matrix[j,:].T
                if eta >= 0: 
                    print("eta>=0")
                    continue
                #步骤4：更新alpha_j
                alphas[j] -= label_vec[j]*(Ei - Ej)/eta
                #步骤5：修剪alpha_j
                alphas[j] = clip_alpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): 
                    print("alpha_j变化太小")
                    continue
                #步骤6：更新alpha_i
                alphas[i] += label_vec[j]*label_vec[i]*(alphaJold - alphas[j])
                #步骤7：更新b_1和b_2
                b1 = b - Ei- label_vec[i]*(alphas[i]-alphaIold)*data_matrix[i,:]*data_matrix[i,:].T - label_vec[j]*(alphas[j]-alphaJold)*data_matrix[i,:]*data_matrix[j,:].T
                b2 = b - Ej- label_vec[i]*(alphas[i]-alphaIold)*data_matrix[i,:]*data_matrix[j,:].T - label_vec[j]*(alphas[j]-alphaJold)*data_matrix[j,:]*data_matrix[j,:].T
                #步骤8：根据b_1和b_2更新b
                if (0 < alphas[i]) and (c > alphas[i]): 
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]): 
                    b = b2
                else: 
                    b = (b1 + b2)/2.0
                #统计优化次数
                alphaPairsChanged += 1
                #打印统计信息
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (iter_num,i,alphaPairsChanged))
        #更新迭代次数
        if (alphaPairsChanged == 0): iter_num += 1
        else: iter_num = 0
        print("迭代次数: %d" % iter_num)
    return b,alphas


def clac_w(dataset, labels, alphas):
    """
    分类器
    :param dataset - 数据集 m*n
    :return w - 权重列表
    """
    alphas, data_mat, label_mat = np.array(alphas), np.array(dataset), np.array(labels)
    w = np.dot((np.tile(label_mat.reshape(1, -1).T, (1, 2)) * data_mat).T, alphas)
    return w.tolist()


def plot_data(dataset, labels, w, b, alphas):
    """
    绘制数据
    :param dataset - 数据集 m*n
    :param w - 
    :param b - 
    :return w - 权重向量
    """
    data_mat = np.mat(dataset)
    # 绘制样本点
    data_plus = []  #正样本
    data_minus = [] #负样本
    for i in range(len(data_mat)):
        if labels[i] > 0:
            data_plus.append(data_mat[i])
        else:
            data_minus.append(data_mat[i])
    data_plus_np = np.array(data_plus)              #转换为numpy矩阵
    data_minus_np = np.array(data_minus)            #转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)   #正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7) #负样本散点图
    # 绘制分类直线 wx+b=0
    x1 = data_mat.max()
    x2 = data_mat.min()
    a1 = float(w[0][0])
    a2 = float(w[1][0])
    b = float(b)
    y1, y2 = (-b- a1*x1)/a2, (-b - a1*x2)/a2
    plt.plot([x1, x2], [y1, y2])
    # 找支持向量点
    sv_x_list = []
    sv_y_list = []
    for i, alpha in enumerate(alphas):
        if abs(alpha) > 0:
            x, y = data_mat[i].tolist()[0]
            sv_x_list.append(x)
            sv_y_list.append(y)
    plt.scatter(sv_x_list, sv_y_list, s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
    plt.show() 