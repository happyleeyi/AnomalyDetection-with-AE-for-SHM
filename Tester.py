import torch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class test_model:
    def __init__(self, net, test_loader, train_loader, num_class, threshold_quantile, device):
        self.net = net
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.num_class = num_class
        self.threshold_quantile = threshold_quantile
        self.device = device

    def cal_loss(self):
        self.net.eval()

        original = [[] for i in range(self.num_class)]
        reconstruct = [[] for i in range(self.num_class)]
        original_normal = []
        reconstruct_normal = []
        label = []
        with torch.no_grad():
            for x, _ in self.train_loader:
                x = x.to(self.device)

                if(len(reconstruct_normal)==0):
                    original_normal = x.clone().data.cpu().numpy()
                    reconstruct_normal = self.net(x).clone().data.cpu().numpy()
                else:
                    original_normal = np.append(original_normal, x.clone().data.cpu().numpy(), axis = 0)
                    reconstruct_normal = np.append(reconstruct_normal, self.net(x).clone().data.cpu().numpy(), axis = 0)

            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if(len(original[0])==0):
                    label = y.clone().data.cpu().numpy()
                    for i in range(self.num_class):
                        original[i] = x[:,:,:,(2-i)*8:(3-i)*8].clone().data.cpu().numpy()
                        reconstruct[i] = self.net(x[:,:,:,(2-i)*8:(3-i)*8]).clone().data.cpu().numpy()
                else:
                    label = np.append(label, y.clone().data.cpu().numpy(), axis=0)
                    for i in range(self.num_class):
                        original[i] = np.append(original[i], x[:,:,:,(2-i)*8:(3-i)*8].clone().data.cpu().numpy(), axis = 0)
                        reconstruct[i] = np.append(reconstruct[i], self.net(x[:,:,:,(2-i)*8:(3-i)*8]).clone().data.cpu().numpy(), axis = 0)
        reconstruct = np.array(reconstruct)
        original = np.array(original)

        original = np.squeeze(original)
        reconstruct = np.squeeze(reconstruct)
        original_normal = np.squeeze(original_normal)
        reconstruct_normal = np.squeeze(reconstruct_normal)

        mse_test = np.mean(np.power(original-reconstruct,2), axis = (2,3))
        mse_train = np.mean(np.power(original_normal-reconstruct_normal,2), axis = (1,2))

        plt.figure(figsize=(15,6))
        
        for i in range(self.num_class):
            print(i)
            ax = plt.subplot(1,self.num_class,i+1)
            ax.set_xlim((0,0.0004))
            ax.set_xticks([0,0.0001,0.0002,0.0003,0.0004])
            ax.hist(mse_train, bins=50, density=True, label="normal", alpha=.6, color="g")
            ax.hist(mse_test[i], bins=50, density=True, label="abnormal", alpha=.6, color="r")

        plt.suptitle("Distribution of the Reconstruction Loss")
        plt.legend()
        plt.savefig('Distribution of the Reconstruction Loss'+'.png')
        plt.clf()

        return mse_train, mse_test, label
    
    def result(self):
        mse_train, mse_test, label = self.cal_loss()
        
        truevalue_test = label
        predict_test = []

        threshold = np.max(mse_train)*self.threshold_quantile
        mse_test[mse_test<threshold] = 0

        mse_test_sum = np.sum(mse_test, axis=0)
        mse_test_argmax = np.argmax(mse_test, axis=0)

        for i in range(label.shape[0]):
            if mse_test_sum[i] == 0:
                predict_test.append(1)
            else:
                predict_test.append(mse_test_argmax[i]+2)

        return predict_test, truevalue_test
    
    def confusion_mat(self):
        predict_test, truevalue_test = self.result()
        #class_names = ['normal', '1f dam', '2f dam', '3f dam']
        matrix1 = confusion_matrix(truevalue_test, predict_test)

        dataframe1 = pd.DataFrame(matrix1)
        plt.figure(figsize=(6,6))
        sns.heatmap(dataframe1, annot=True, cbar=None, cmap="Blues")
        plt.title("Confusion Matrix_test"), plt.tight_layout()
        plt.ylabel("True Class"), plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig('confusion matrix with autoencoder method'+'('+str(self.threshold_quantile)+')'+'.png')
        plt.clf()

        accuracy = np.sum(predict_test==truevalue_test)/len(predict_test)
        print("accuracy : ", accuracy)

        f = open("record.txt", "a+")
        f.write("threshold "+str(self.threshold_quantile)+" accuracy : %f\n" %accuracy)
        f.close()



        