import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    # 从banking.csv文件中读取数据
    data = pd.read_csv(r'./banking.csv', sep=',')
    # 数据预处理
    for i in data.keys():
        new_columns = {}
        index = 0
        # age
        if i == 'age':
            for j in data.index:
                if data.loc[j, i] in range(40, 60):
                    data.loc[j, i] = 3
                elif data.loc[j, i] in range(30, 40):
                    data.loc[j, i] = 2
                elif data.loc[j, i] in range(20, 30):
                    data.loc[j, i] = 1
                else:
                    data.loc[j, i] = 0
        elif i == 'job':
            for j in data.index:
                if data.loc[j, i] in ['admin.', 'entrepreneur', 'management', 'technician']:
                    data.loc[j, i] = 2
                elif data.loc[j, i] in ['blue-collar', 'housemaid', 'retired', 'self-employed', 'services', 'student',
                                        'unemployed']:
                    data.loc[j, i] = 1
                else:
                    data.loc[j, i] = 0
        elif i == 'marital':
            for j in data.index:
                if data.loc[j, i] == 'married':
                    data.loc[j, i] = 3
                elif data.loc[j, i] == 'divorced':
                    data.loc[j, i] = 2
                elif data.loc[j, i] == 'single':
                    data.loc[j, i] = 1
                else:
                    data.loc[j, i] = 0
        elif i == 'education':
            for j in data.index:
                if data.loc[j, i] == 'professional.course':
                    data.loc[j, i] = 3
                elif data.loc[j, i] == 'university.degree':
                    data.loc[j, i] = 2
                elif data.loc[j, i] in ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate']:
                    data.loc[j, i] = 1
                else:
                    data.loc[j, i] = 0
        elif i == 'pdays':
            for j in data.index:
                if not data.loc[j, i] == 999:
                    data.loc[j, i] = 1
                else:
                    data.loc[j, i] = 0
        elif data[i].dtype == 'object':
            for j in data.index:
                key = data.loc[j, i]
                if key not in new_columns.keys():
                    new_columns[key] = index
                    index += 1
            data[i] = data[i].map(new_columns)
    # 将特征值与标签'y'分离
    feature = data.drop('y', 1).values
    label = data['y'].values
    # 特征值归一化
    feature = scale(feature)

    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.2,
                                                                            random_state=40)

    lg = LogisticRegression()
    model = lg.fit(feature_train, label_train)
    prediction = model.predict(feature_test)
    pre = precision_score(label_test, prediction)
    acc = accuracy_score(label_test, prediction)
    rec = recall_score(label_test, prediction)
    f1 = f1_score(label_test, prediction)
    # s = lg.score(feature_test, label_test)
    print(pre, acc, rec, f1)

    # # 使用6折交叉验证，并且画ROC曲线
    # skf = StratifiedKFold(n_splits=6)
    # lg = LogisticRegression()
    #
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)
    # all_tpr = []
    # k = 0
    #
    # for train, test in skf.split(feature, label):
    #     lg.fit(feature[train], label[train])
    #     predicts = lg.predict_proba(feature[test])
    #     fpr, tpr, thresholds = roc_curve(label[test], predicts[:, 1], pos_label=1)
    #     roc_auc = auc(fpr, tpr)
    #     # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    #     plt.plot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (k, roc_auc,))
    #     k += 1
    #     # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    #     mean_tpr += np.interp(mean_fpr, fpr, tpr)
    # mean_tpr[0] = 0.0  # 初始处为0
    # mean_tpr /= 6  # 在mean_fpr100个点，每个点处插值插值多次取平均
    # mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    # mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    # # 画平均ROC曲线
    # plt.plot(mean_fpr, mean_tpr, 'k--',
    #          label='Mean ROC (AUC = %0.2f)' % mean_auc, lw=2)
    #
    # # 画对角线
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #
    # # 图例属性设置
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # # # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()
