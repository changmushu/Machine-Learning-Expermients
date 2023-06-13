# Machine-Learning-Expermients

学习通课后习题 https://github.com/changmushu/Machine-Learning-Expermients/blob/master/Exam/Promble.md

### 东北林业大学-机器学习实验

#### 注意：使用前需要将数据集路径修改，不可直接运行。

#### 实验一：

（一）根据给定数据集，利用线性回归和多项式回归模型训练和测试一个数据预测模型，并对模型的性能和预测能力进行分析；

要求：

（1）线性回归，用1次曲线，2次曲线，并画出曲线观察；

（2）用多项式回归，并画出多项式次数n取不同值时的拟合曲线，并观察分析n取多少时，模型效果最好。

（二）利用线性回归模型进行波斯顿房价预测

利用马萨诸塞州波士顿郊区的房屋信息数据，利用线性回归模型训练和测试一个房价预测模型，并对模型的性能和预测能力进行测试分析；

要求：（1）画出误差变化曲线图；（2）画出预测值与真实值的对应变化曲线并分析

（三）利用逻辑回归模型进行莺尾花分类预测

（四）利用逻辑回归模型进行心脏病预测

心脏病是人类健康的头号杀手。全世界1／3的人口死亡是因心脏病引起的，而我国，每年有几十万人死于心脏病。 所以，如果可以通过提取人体相关的体侧指标，通过数据挖掘的方式来分析不同特征对于心脏病的影响，对于预测和预防心脏病将起到至关重要的作用。

要求：（1）画出混淆矩阵；（2）画出代价函数变化趋势曲线 ；（3）画出ROC曲线，并分析模型特性。

#### 实验二：

SVM向量机
 使用SMO算法设计实现SVM的分类算法：
   （1）给定一个有两个特征、包含两种类别数据集testSet.txt，然后用线性核函数的SVM算法进行分类；最后把SVM算法拟合出来的分隔超平面画出来。
   （2）构建一个对非线性可分的数据集testSetRBF.txt进行有效分类的分类器，该分类器使用了径向基核函数K(x,y)如下所示。
 
其中，σ是用户定义的用于确定到达率(reach)或者说函数值跌落到0的速度参数。
    上述高斯核函数将数据从其特征空间映射到更高维的空问，具体来说这里是映射到一个无穷维的空问。
首先，需要确定σ的大小，然后利用该核函数构建出一个分类器。
你可以尝试更换不同的σ参数以观察测试错误率、训练错误率、支持向量个数随k1的变化情况。例如给出当σ=0.1，或σ=1.3时的结果。最后把不同σ的SVM算法拟合出来的分隔超平面画出来。通过修改模型参数σ的值，可以调整分隔超平面的形状，观察结果有什么变化。比如σ值太大，太小对分隔效果有什么影响。
（3）网易财经上获得的上证指数的历史数据，爬取了20年的上证指数数据。根据给出当前时间前150天的历史数据，预测当天上证指数的涨跌。用sklearn提供的包完成。
（4）（选做）用支持向量机模型设计实现手写数字识别问题。可以用scklearn提供的包完成。
