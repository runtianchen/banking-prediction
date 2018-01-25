# Attribute Information:
#
# Input variables:
# # bank client data:
# 1 - age (numeric)
# 2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# 3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# 4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# 5 - default: has credit in default? (categorical: 'no','yes','unknown')
# 6 - housing: has housing loan? (categorical: 'no','yes','unknown')
# 7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# # related with the last contact of the current campaign:
# 8 - contact: contact communication type (categorical: 'cellular','telephone')
# 9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# 10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# # other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)
# 15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# # social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
# 20 - nr.employed: number of employees - quarterly indicator (numeric)
#
# Output variable (desired target):
# 21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":
    # 从banking.csv文件中读取数据
    data = pd.read_csv(r'./banking.csv', sep=',')
    # 将所有非数值型特征转换为数值型
    for i in data.keys():
        new_columns = {}
        index = 0
        if data[i].dtype == 'object':
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

    # 使用6折交叉验证，并且画ROC曲线
    skf = StratifiedKFold(n_splits=6)
    lg = LogisticRegression()

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    k = 0

    for train, test in skf.split(feature, label):
        lg.fit(feature[train], label[train])
        predicts = lg.predict_proba(feature[test])
        fpr, tpr, thresholds = roc_curve(label[test], predicts[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (k, roc_auc,))
        k += 1
        # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0  # 初始处为0
    mean_tpr /= 6  # 在mean_fpr100个点，每个点处插值插值多次取平均
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    # 画平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='Mean ROC (AUC = %0.2f)' % mean_auc, lw=2)

    # 画对角线
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

    # 图例属性设置
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
