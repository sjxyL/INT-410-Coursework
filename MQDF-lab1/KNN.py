import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

#calculate distance function
def cal_distance(test,cal):
    label_distance_list=[]#list of label and distance
    for i in range(len(cal)):
        label_distance=[]
        label_distance.append(cal[i, 4])#append corresponding label
        d=test-cal[i,:] #test sample value-training sample value
        label_distance.append(np.linalg.norm(d)) #calculate distance
        label_distance_list.append(label_distance)
    return label_distance_list

if __name__=='__main__':
    iris_data=pd.read_csv("iris.data",header=None)
    labels_codes=pd.Categorical(iris_data[4]).codes
    for i in range(150):
        iris_data.loc[i,4]=labels_codes[i]
    datalist=iris_data.values.tolist()
    random.seed(17)
    random.shuffle(datalist)
    data_set=np.mat(datalist)

    average_acc=[]
    k_list=[]
    #  test different K values:
    for K in range(1, 120):
        if K%3!=0:
            # for visualization
            k_list.append(K)
            accuracy = []
            #implementing 5-fold training process:
            length = len(data_set) // 5
            for i in range(5):
                #cal_data, test_data =  # split training and testing set
                test_data = data_set[i * length:(i + 1) * length]  # Select the i-th fold as the test set
                cal_data = np.concatenate([data_set[j * length:(j + 1) * length] for j in range(5) if j != i])  # Combine the rest as the training set
                # doing KNN
                right = 0
                for x in range(len(test_data)):
                    d_set = np.mat(cal_distance(test_data[x], cal_data))#list of distance
                    d_set = (d_set[np.lexsort(d_set.T)])[0, :, :]#sort distance list
                    p_wk = [0, 0, 0] #be used to record P(wk|x)
                    #calculate P(wk|x) for each test sample
                    for y in range(K):
                        if d_set[y, 0] == 0:
                            p_wk[0] = p_wk[0] + 1 / K
                        elif d_set[y, 0] == 1:
                            p_wk[1] = p_wk[1] + 1 / K
                        else:
                            p_wk[2] = p_wk[2] + 1 / K
                    #calculate accuracy
                    if p_wk.index(max(p_wk)) == test_data[x, 4]:
                        right = right + 1
                accuracy.append(right / len(test_data))
            accuracy=np.mat(accuracy)
            # print(accuracy)
            average_acc.append(np.mean(accuracy))

    plt.scatter(k_list,average_acc)
    plt.title('KNN')
    plt.xlabel('hyperparameter K')
    plt.ylabel('average accuracy')
    plt.show()
    k_max_list=[]
    for z in range(len(average_acc)):
        if average_acc[z]==max(average_acc):
            k_max_list.append(k_list[z])
    print("highest average accuracy:",round(max(average_acc),3))
    print("corresponding K:",k_max_list)
