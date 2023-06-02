from sklearn import datasets
import matplotlib.pyplot as plt

digits=datasets.load_digits()
#
images_and_labels=list(zip(digits.images,digits.target))
plt.figure(figsize=(8,6),dpi=200)
for index,(image,label) in enumerate(images_and_labels[:8]):
    plt.subplot(2,4,index+1)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Digit:%i'%label,fontsize=20)
#plt.show()
print(digits.images.shape)
print(digits.images.data.shape)
print(digits.target.shape)

#把数据集分成训练集和测试集
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(digits.data,digits.target,test_size=0.2,random_state=2)
from sklearn import svm

clf=svm.SVC(gamma=0.001,C=100.)
clf.fit(Xtrain,Ytrain)

print("测试精度为：",clf.score(Xtest,Ytest))
Ypred=clf.predict(Xtest)

#预测情况
fig,axes=plt.subplots(4,4,figsize=(8,8))
fig.subplots_adjust(hspace=0.1,wspace=0.1)

for i,ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8,8),cmap=plt.cm.gray_r,interpolation='nearest')
    ax.text(0.05,0.05,str(Ypred[i]),fontsize=32,transform=ax.transAxes,color='green' if Ytest[i]==Ypred[i] else 'red')
    ax.text(0.8,0.05,str(Ytest[i]),fontsize=32,transform=ax.transAxes,color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

#保存模型参数
import joblib
joblib.dump(clf,'digits_svm.pkl')

#导入模型参数，直接进行预测
clf=joblib.load('digita_svm.pkl')
Ypredd=clf.predict(Xtest)
print("预测精度：",clf.score(Ytest,Ypredd))