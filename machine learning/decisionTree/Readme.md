@[TOC]
# 1.基本思想
## 1.1 基本思想
建立分段函数进行回归和分类，自动学一个if-else-then的模型.对于连续数据，学习机或者模型的函数集合来自于平行于坐标轴的直线，对于离散数据，学习机就是单独的特征，选择单独的特征作为节点，由离散值的种类确定分支数。
西瓜书根据西瓜的一些特征来判断西瓜的好坏，而一次我们只用一个特征，或者只看一个维度，那么根据数据集，我们可以根据一些指标(这个指标指导每次如何选择特征，如信息增益等)使得我们可以更快更准的判断出一个西瓜是不是好瓜(做决策)，这个推断过程类似一个树性结构，于是算法起名决策树。我们根据如下数据集
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119083337782.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
参考[深入浅出理解决策树算法（一）-核心思想](https://zhuanlan.zhihu.com/p/26703300)
[西瓜书](https://github.com/datawhalechina/pumpkin-book)
得到的树形结构为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118221157152.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
决策树模型核心是下面几部分：

* 结点和对应的分支
* 结点有内部结点和叶结点，内部结点表示一个特征，叶节点表示一个类

# 2.预测机理
机器学习不用于统计学在于，[机器学习重在预测，统计学重在研究变量关系](https://www.jiqizhixin.com/articles/2019-05-06-13)，但是二者由很多相似之处。

[如果树建立起来了，假如我现在告诉你，我买了一个西瓜，它的特点是纹理是清晰，根蒂是硬挺的瓜，你来给我判断一下是好瓜还是坏瓜，恰好，你构建了一颗决策树，告诉他，没问题，我马上告诉你是好瓜，还是坏瓜？](https://zhuanlan.zhihu.com/p/26703300)

判断步骤如下：

根据纹理特征，已知是清晰，那么走下面这条路，红色标记：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118221530670.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)好的，现在到了第二层了，这个时候，我们需要知道根蒂的特征是什么？很好，输入的西瓜根蒂是硬挺，于是，我们继续走，如下面蓝色所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118221600989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
此时，我们到达叶子结点了，根据上面总结的点，可知，叶子结点代表一种类别，我们从如上决策树中，可以知道，这是一个坏瓜！

写成数学公式就是 $\hat f$(x) = $\sum\limits_{m=1}^9 c_m \delta(\bm x)$, 9个叶子节点所以为m从0到9，9种决策，但是9决策不一定代表西瓜的种类有九种。
每一个从根节点到叶子结点路径上特征及其取值的集合x对应的$\delta(x)$函数取值为1，比如本题,$\delta(x=纹理清晰，根蒂 =硬挺，触感=*，色泽 =* ) = 1$,其余的$\delta(x)$取值为0.
# 3.划分选择
[ID3、C4.5、CART三种决策树的区别](https://blog.csdn.net/qq_27717921/article/details/74784400)，
###  ID3
假设样本集合D中第k类样本所占的比重为$p_k$，那么信息熵的计算则为下面的计算方式

$$Ent(D) = - \sum\limits_{k=1}^{|y|}p_klog_2p_k$$
当这个Ent(D)的值越小，说明样本集合D的纯度就越高
有了信息熵，当我选择用样本的某一个属性a来划分样本集合D时，就可以得出用属性a对样本D进行划分所带来的“信息增益”$$
Gain(D,a) = Ent(D)  - \sum\limits_{v=1}^{|V|}\frac{D^v}{D}Ent(D^v)$$ 
V是属性的可能取值。$y= -p_klog_2p_k$的函数曲线图如图所示，分类越多，p越小，y的值越小，许多很小的加权求和起来也不大，使得Gain(D,a) 很大。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119090336826.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
```python
from matplotlib import pyplot as plt
import  numpy as np
p= np.linspace(0.01, 1.0001, 5000)
Ent =  -p*np.log2(p)
plt.plot(p,Ent)
plt.title("-p*log2(p) curve graph")
plt.xlabel("p")
plt.ylabel("-p*log2(p)")
plt.show()
```
<font color=Brown size=3 face="微软雅黑">ID3以信息增益作为结点的划分依据，容易偏向与选择分支数比较多的特征或者说属性</font>，极端情况就是选择数据集中的编号更容易最大信息增益。
### c4.5
 C4.5使用信息增益率,信息增益除以固有值(intrinsic value)
 $$Gain\_ratio (D,a) =  \frac{Gain(D,a)}{IV(a)}$$
 $$IV(a) = - \sum\limits_{v=1}^{V}\frac{D^v}{D}log_2(\frac{D^v}{D})$$
 但是同样的<font color=Brown size=3 face="微软雅黑">c4.5的增益率对可取值数目较少的属性有所偏好</font>，因此C4.5决策树先从候选划分属性中找出信息增益高于平均水平的属性，在从中选择增益率最高的。
### CART决策树
#### 对于分类问题
CART决策树使用“基尼指数”（Gini index）来选择划分属性，基尼指数反映的是从样本集D中随机抽取两个样本，其类别标记不一致的概率，因此Gini(D)越小越好，基尼指数定义如下:
$$ Gini(D) = \sum\limits_{k=1}^{|y|}\sum\limits_{k' \neq k} p_kp_k' = \sum\limits_{k=1}^{|y|}p_k(1-p_k) =1-  \sum\limits_{k=1}^{|y|}p^2_k $$
进而，使用属性α划分后的基尼指数为：
$$Gini_index (D,a) = \sum\limits_{v=1}^{V} \frac{D^v}{D}Gini(D^v)$$
#### 对于回归问题
最小二乘误差$$\min\limits_{\hat f}\sum\limits_{i=1}^m(\hat f(x_i)-y_i)^2$$
其中$\hat f(x)$是决策树的闭式表达式，y是真值，<font color=Brown size=3 face="微软雅黑">least square error其实事先假定了数据点服从高斯分布。</font>

# 4. 基本流程
根据loss function选择特征、生成决策树、对决策树进行修剪。
### 特征选择
特征选择决定了使用哪些特征来做判断。在训练数据集中，每个样本的属性可能有很多个，不同属性的作用有大有小。因而特征选择的作用就是筛选出跟分类结果相关性较高的特征，也就是分类能力较强的特征。
在特征选择中通常使用的准则是：信息增益。

### 决策树生成
选择好特征后，就从根节点触发，对节点计算所有特征的信息增益，选择信息增益最大的特征作为节点特征，根据该特征的不同取值建立子节点；对每个子节点使用相同的方式生成新的子节点，直到信息增益很小或者没有特征可以选择为止。
### 决策树剪枝
剪枝的主要目的是对抗「过拟合」，通过主动去掉部分分支来降低过拟合的风险。
也分预剪枝和后剪枝
# 4. 分类与回归例子
## 分类
以鸢尾花数据集合为例,数据集合有150个数据,特征包含花瓣长度，宽度，花萼长度，宽度总共四个数据，类别为'setosa', 'versicolor', 'virginica'三种。为了可视化我们选取两个属性。
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier,plot_tree


# 使用自带的iris数据，为了方便可视化，我们能使用前两维度
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# 限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)
# 决策树
plt.figure()
plot_tree(clf, filled=True)
plt.show()
plt.figure()

# 画图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z,cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```
结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111915475944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
可以看到对于连续值，建立的树为二叉树，因为根据特征值是都大于某个数将数据集合一分为2，为了限制过拟合，我们设计了最大深度，也可以利用剪枝等操作防止过拟合。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119154808616.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
决策树得到的分类面如图所示，用平行于坐标轴的直线，组合出分界面。
- 易于理解和解释，甚至比线性回归更直观；
- 与人类做决策思考的思维习惯契合；
- 模型可以通过树的形式进行可视化展示；

## 回归
我们根据y=ax+b随机产生80个点，加一点噪声，使用MSE，限制最大深度，得到的决策树为：
```python
print(__doc__)

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
a = 5
b = 2
y = (a*X+b).ravel()
y[::5] += 3 * (0.9 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
```
结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119160616612.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
对于二次函数的拟合
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119161404528.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
将第二个数据点进行改动
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119180649278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
因为决策树的分类面是个直线，因此它也是用常数函数去拟合二次函数，得到的效果自然不是很好。在特定维度函数非线性的程度很大的话，很难用决策树去拟合。

# 5. 过拟合与正则化

奥卡姆剃刀原理：越简单越好。
no free lunch 定理：所有算法的期望性能相同，
避免Regression Tree的过拟合策略
 * 检测目标函数值下降小于一定阈值时终止
* 当树达到一定预设高度上界时终止
* 当每个区域包含点个数少于某个预设阈值
时终止
* 控制深度
* <font color=Brown size=3 face="微软雅黑">剪枝 </font>
*  <font color=Brown size=3 face="微软雅黑">控制策略嵌入优化模型：正则化</font>
* <font color=Brown size=3 face="微软雅黑">构造验证集</font>
在验证集效果提升的前提下生长树
再生成过拟合完全数后，在验证集效果
提升的前提下切割树

## 正则化
控制策略嵌入优化模型：正则化
$$\min\limits_{\hat f}\sum\limits_{i=1}^n (\hat f(x_i)-y_i) + \alpha|\hat f|$$
在树生长时，前一项（原目标函数）不断减小，后一项（正则项）不断增加, 也就是模型复杂度在增加。
# 6. 缺失值处理
缺失值处理和多变量决策具体参见西瓜书。
（1）如何选择划分属性。（2）给定划分属性，若某样本在该属性上缺失值，如何划分到具体的分支上
* 对于（1）：通过在样本集D中选取在属性α上没有缺失值的样本子集，计算在该样本子集上的信息增益，最终的信息增益等于该样本子集划分后信息增益乘以样本子集占样本集的比重。即：
* 对于（2）：若该样本子集在属性α上的值缺失，则将该样本以不同的权重（即每个分支所含样本比例）划入到所有分支节点中。
# 7. 多变量决策
学习机不再是平行于坐标轴的线，而是有多个属性进行线性组合.
# 8. Question and answer
#### Ⅰ 什么是机器学习？
"A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E"
机器学习三要素
* 使用先验知识和经验数据 (Experience)
* 针对某个目标 (Task)
*  提高学习性能 (Performance）

从三方面理解机器学习
要学什么   ->    决策函数
从哪里学   ->    训练数据+定理
怎样学习  ->      求解机器学习模型，得到mapping

不同角度理解机器学习

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119170640542.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
#### Ⅱ 机器学习的根本要求
机器学习的根本要求是做<font color=Red size=3 face="微软雅黑">预测和防止过拟合</font>, 预测分类，预测连续的一个数值等。但在训练模型的过程中要防止过拟合

#### Ⅲ 模型驱动与数据驱动的区别
* 模型驱动可以理解为知识驱动，结合数据和一些定理，规则进行学习，求解模型，模型的可解释性好。
* 数据驱动, 假设所有信息都可以来源于数据，大数据时代，数据量远远大于模型参数量，我们可以只利用数据建立预测模型神行预测，指深度学习。
在之前的文章[A Neural Model for Generating Natural Language Summaries of Program Subroutines](https://arxiv.org/abs/1902.01954)中,
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191119172109885.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MDk3Mzkz,size_16,color_FFFFFF,t_70)
数据集：2.1 million
参数数量：159million
这个问题可能会导致模型是欠拟合，在试验中。

#### Ⅳ决策树的思想
* 数学理解是建立分段函数，程序理解是通过数据，在一定规则下，得到if- else- then的树形判别结构。
* 虽然和if-else没区别，但是if what then what是从数据中基于规则学出来的，而不再需要对于某个问题去自己设计。
* 决策树的好坏一方面取决于数据，另一方面取决于设计的特征。


#### Ⅴ 决策树的优缺点
优点

- 易于理解和解释，甚至比线性回归更直观；
- 与人类做决策思考的思维习惯契合，有时候不需要获得全部信息就可以获得决策结果；
- 模型可以通过树的形式进行可视化展示；
- 可以直接处理非数值型数据
- 
缺点
- 模型不够稳健，因为它使用横竖线找分类面或者拟合数据，而且充分考虑到了每一个数据，一个点的变化导致树的结构发生变化。
- 数值型变量之间存在许多错综复杂的关系，互相不解耦合的话，很难用直线来分
- 随着数据量的增大，树会建立的很大，很深。

参考文献
[决策树 – Decision tree](https://easyai.tech/ai-definition/decision-tree/)
[深入浅出理解决策树算法（一）-核心思想](https://zhuanlan.zhihu.com/p/26703300)
[Vay-keen/Machine-learning-learning-notes](https://github.com/Vay-keen/Machine-learning-learning-notes/blob/master/%E5%91%A8%E5%BF%97%E5%8D%8E%E3%80%8AMachine%20Learning%E3%80%8B%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0(5)--%E5%86%B3%E7%AD%96%E6%A0%91.md)
[sklearn](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
[决策树算法的Python实现](https://zhuanlan.zhihu.com/p/20794583)
[一文读懂统计学与机器学习的本质区别（附案例）](https://www.jiqizhixin.com/articles/2019-05-06-13)


