## 目录
* [机器学习算法Python实现](#机器学习算法python实现)
	* [一、线性回归](#一线性回归)
		* [1、代价函数](#1代价函数)
		* [2、梯度下降算法](#2梯度下降算法)
		* [3、均值归一化](#3均值归一化)
		* [4、最终运行结果](#4最终运行结果)
		* [5、使用scikit-learn库中的线性模型实现](#5使用scikit-learn库中的线性模型实现)
	* [二、逻辑回归](#二逻辑回归)
		* [1、代价函数](#1代价函数)
		* [2、梯度](#2梯度)
		* [3、正则化](#3正则化)
		* [4、S型函数（即）](#4s型函数即)
		* [5、映射为多项式](#5映射为多项式)
		* [6、使用的优化方法](#6使用scipy的优化方法)
		* [7、运行结果](#7运行结果)
		* [8、使用scikit-learn库中的逻辑回归模型实现](#8使用scikit-learn库中的逻辑回归模型实现)
	* [逻辑回归_手写数字识别_OneVsAll](#逻辑回归_手写数字识别_onevsall)
		* [1、随机显示100个数字](#1随机显示100个数字)
		* [2、OneVsAll](#2onevsall)
		* [3、手写数字识别](#3手写数字识别)
		* [4、预测](#4预测)
		* [5、运行结果](#5运行结果)
		* [6、使用scikit-learn库中的逻辑回归模型实现](#6使用scikit-learn库中的逻辑回归模型实现)
	* [三、BP神经网络](#三bp神经网络)
		* [1、神经网络model](#1神经网络model)
		* [2、代价函数](#2代价函数)
		* [3、正则化](#3正则化)
		* [4、反向传播BP](#4反向传播bp)
		* [5、BP可以求梯度的原因](#5bp可以求梯度的原因)
		* [6、梯度检查](#6梯度检查)
		* [7、权重的随机初始化](#7权重的随机初始化)
		* [8、预测](#8预测)
		* [9、输出结果](#9输出结果)
	* [四、SVM支持向量机](#四svm支持向量机)
		* [1、代价函数](#1代价函数)
		* [2、Large Margin](#2large-margin)
		* [3、SVM Kernel（核函数）](#3svm-kernel核函数)
		* [4、使用中的模型代码](#4使用scikit-learn中的svm模型代码)
		* [5、运行结果](#5运行结果)
	* [五、K-Means聚类算法](#五k-means聚类算法)
		* [1、聚类过程](#1聚类过程)
		* [2、目标函数](#2目标函数)
		* [3、聚类中心的选择](#3聚类中心的选择)
		* [4、聚类个数K的选择](#4聚类个数k的选择)
		* [5、应用——图片压缩](#5应用图片压缩)
		* [6、使用scikit-learn库中的线性模型实现聚类](#6使用scikit-learn库中的线性模型实现聚类)
		* [7、运行结果](#7运行结果)
	* [六、PCA主成分分析（降维）](#六pca主成分分析降维)
		* [1、用处](#1用处)
		* [2、2D-->1D，nD-->kD](#22d--1dnd--kd)
		* [3、主成分分析PCA与线性回归的区别](#3主成分分析pca与线性回归的区别)
		* [4、PCA降维过程](#4pca降维过程)
		* [5、数据恢复](#5数据恢复)
		* [6、主成分个数的选择（即要降的维度）](#6主成分个数的选择即要降的维度)
		* [7、使用建议](#7使用建议)
		* [8、运行结果](#8运行结果)
		* [9、使用scikit-learn库中的PCA实现降维](#9使用scikit-learn库中的pca实现降维)
	* [七、异常检测 Anomaly Detection](#七异常检测-anomaly-detection)
		* [1、高斯分布（正态分布）](#1高斯分布正态分布gaussian-distribution)
		* [2、异常检测算法](#2异常检测算法)
		* [3、评价的好坏，以及的选取](#3评价px的好坏以及ε的选取)
		* [4、选择使用什么样的feature（单元高斯分布）](#4选择使用什么样的feature单元高斯分布)
		* [5、多元高斯分布](#5多元高斯分布)
		* [6、单元和多元高斯分布特点](#6单元和多元高斯分布特点)
		* [7、程序运行结果](#7程序运行结果)

# 一、线性回归
 - 全部代码
### 1、代价函数
- J(\theta ) = \frac{1}{{2{\text{m}}}}\sum\limits_{i = 1}^m {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}} <br>
- 其中：{h_\theta }(x) = {\theta _0} + {\theta _1}{x_1} + {\theta _2}{x_2} + ...<br>
- 下面就是要求出theta，使代价最小，即代表我们拟合出来的方程距离真实值最近<br>
- 共有m条数据，其中 {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}}<br>
- 代表我们要拟合出来的方程到真实值距离的平方，平方的原因是因为可能有负值，正负可能会抵消<br>
- 前面有系数2的原因是下面求梯度是对每个变量求偏导，2可以消去<br>
- 实现代码：<br>
```
# 计算代价函数
import numpy as np

def computeCost(X, y, theta):
    m = len(y)
    errors = X @ theta - y  # 矩阵乘法，得到预测误差
    cost = (errors.T @ errors) / (2 * m)
    return cost.item()  # 提取标量

```
- 注意这里的X是真实数据前加了一列1，因为有theta(0)<br>

### 2、梯度下降算法 
- 代价函数对{{\theta _j}}求偏导得到：
  \frac{{\partial J(\theta )}}{{\partial {\theta j}}} = \frac{1}{m}\sum\limits{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]}
- 所以对theta的更新可以写为：
  {\theta j} = {\theta j} - \alpha \frac{1}{m}\sum\limits{i = 1}^m {[({h\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]}
- 其中\alpha为学习速率，控制梯度下降的速度，一般取0.01，0.03，0.1，0.3....
- 为什么梯度下降可以逐步减小代价函数
- 假设函数f(x)
- 泰勒展开：f(x+△x)=f(x)+f'(x)*△x+o(△x)
- 令：△x=-α*f'(x) ,即负梯度方向乘以一个很小的步长α
- 将△x代入泰勒展开式中：f(x+△x)=f(x)-α*[f'(x)]²+o(△x)
- 可以看出，α是取得很小的正数，[f'(x)]²也是正数，所以可以得出：f(x+△x)<=f(x)
- 所以沿着负梯度放下，函数在减小，多维情况一样
- 实现代码
```
 # 梯度下降算法 
def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)      
    n = len(theta)
    
    temp = np.matrix(np.zeros((n,num_iters)))   # 暂存每次迭代计算的theta，转化为矩阵形式
    
    
    J_history = np.zeros((num_iters,1)) #记录每次迭代计算的代价值
    
    for i in range(num_iters):  # 遍历迭代次数    
        h = np.dot(X,theta)     # 计算内积，matrix可以直接乘
        temp[:,i] = theta - ((alpha/m)*(np.dot(np.transpose(X),h-y)))   #梯度的计算
        theta = temp[:,i]
        J_history[i] = computerCost(X,y,theta)      #调用计算代价函数
        print '.',      
    return theta,J_history
```
### 3、均值归一化
- 目的是使数据都缩放在一个范围内，便于使用梯度下降算法
- {x_i} = \frac{{{x_i} - {\mu _i}}}{{{s_i}}}
- 其中{{\mu _i}}为所有此feture数据的平均值
- {{s_i}}可以是最大值-最小值，也可以是这个feature对应的数据的标准差
- 实现代码
 ```
  # 归一化feature
def featureNormaliza(X):
    X_norm = np.array(X)            #将X转化为numpy数组对象，才可以进行矩阵的运算
    #定义所需变量
    mu = np.zeros((1,X.shape[1]))   
    sigma = np.zeros((1,X.shape[1]))
    
    mu = np.mean(X_norm,0)          # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm,0)        # 求每一列的标准差
    for i in range(X.shape[1]):     # 遍历列
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]  # 归一化
    
    return X_norm,mu,sigma
 ```
- 注意预测的时候也需要均值归一化数据

  ### 4、最终运行结果
  - 代价随迭代次数的变化<br>
  ![image](https://github.com/user-attachments/assets/f1eb2473-b415-4811-abab-8abbb3593316)

### 5、使用scikit-learn库中的线性模型实现
- 导入包
```
   from sklearn import linear_model
   from sklearn.preprocessing import StandardScaler    #引入缩放的包
```
- 归一化
```
    # 归一化操作
    scaler = StandardScaler()   
    scaler.fit(X)
    x_train = scaler.transform(X)
    x_test = scaler.transform(np.array([1650,3]))
```
- 线性模型拟合
```
    # 线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train, y)
```
- 预测
```
    #预测结果
    result = model.predict(x_test)
```

# 二、逻辑回归
- 全部代码
### 1、代价函数
- ![image](https://github.com/user-attachments/assets/51a24067-be38-47ea-b2b2-787867bb0dd6)
- 可以综合起来为：
- J(\theta ) = - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})] 其中：
- {h_\theta }(x) = \frac{1}{{1 + {e^{ - x}}}}
- 为什么不用线性回归的代价函数表示，因为线性回归的代价函数可能是非凸的，对于分类问题，使用梯度下降很难降到最小值，上面的代价函数是凸函数
- { - \log ({h_\theta }(x))}的图像如下，即y=1时：![image](https://github.com/user-attachments/assets/0257856e-9c00-4788-ae8f-6932dc0ab82c)
- 可以看出，当{{h_\theta }(x)}趋于1，y=1,与预测值一致，此时付出的代价cost趋于0，若{{h_\theta }(x)}趋于0，y=1,此时的代价cost值非常大，我们最终的目的是最小化代价值
- 同理{ - \log (1 - {h_\theta }(x))}的图像如下（y=0）：
- ![image](https://github.com/user-attachments/assets/0bbd27ad-d460-4422-aa78-aaec64a7d973)


### 2、梯度
- 同样对代价函数求偏导：
- \frac{{\partial J(\theta )}}{{\partial {\theta j}}} = \frac{1}{m}\sum\limits{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]}
- 可以看出来与线性回归的偏导数一致
- 推导过程
- ![image](https://github.com/user-attachments/assets/4d7dbfa1-a1d0-4693-9260-e34b1a5c675c)




### 3、正则化
- 目的是为了防止过拟合
- 在代价函数上加上一项
  J(\theta ) = - \frac{1}{m}\sum\limits_{i = 1}^m {[{y^{(i)}}\log ({h_\theta }({x^{(i)}}) + (1 - } {y^{(i)}})\log (1 - {h_\theta }({x^{(i)}})] + \frac{\lambda }{{2m}}\sum\limits_{j = 1}^n {\theta _j^2}
- 注意j是从1开始的，因为theta(0)为一个常数项，X中最前面一列会加上一列1，所以乘积还是theta(0),与feature没有关系，，没有必要正则化
- 正则化的代价
```
# 代价函数
def costFunction(initial_theta,X,y,inital_lambda):
    m = len(y)
    J = 0
    
    h = sigmoid(np.dot(X,initial_theta))    # 计算h(z)
    theta1 = initial_theta.copy()           # 因为正则化j=1从1开始，不包含0，所以复制一份，前theta(0)值为0 
    theta1[0] = 0   
    
    temp = np.dot(np.transpose(theta1),theta1)
    J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+temp*inital_lambda/2)/m   # 正则化的代价方程
    return J
```
- 正则化的代价的梯度
```
# 计算梯度
def gradient(initial_theta,X,y,inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))
    
    h = sigmoid(np.dot(X,initial_theta))# 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X),h-y)/m+inital_lambda/m*theta1 #正则化的梯度
    return grad  
```
### 4、S型函数（即{{h_\theta }(x)}）
- 实现代码
```
# S型函数    
def sigmoid(z):
    h = np.zeros((len(z),1))    # 初始化，与z的长度一置
    
    h = 1.0/(1.0+np.exp(-z))
    return h
```
### 5、映射为多项式
- 因为数据的feture可能很少，导致偏差大，所以创造出一些feture结合
- eg:映射为2次方的形式:1 + {x_1} + {x_2} + x_1^2 + {x_1}{x_2} + x_2^2
- 实现代码：
```
# 映射为多项式 
def mapFeature(X1,X2):
    degree = 3;                     # 映射的最高次方
    out = np.ones((X1.shape[0],1))  # 映射后的结果数组（取代X）
    '''
    这里以degree=2为例，映射为1,x1,x2,x1^2,x1,x2,x2^2
    '''
    for i in np.arange(1,degree+1): 
        for j in range(i+1):
            temp = X1**(i-j)*(X2**j)    #矩阵直接乘相当于matlab中的点乘.*
            out = np.hstack((out, temp.reshape(-1,1)))
    return out
```
### 6、使用scipy的优化方法
- 梯度下降使用scipy中的optimize中的fmin_bfgs函数
- 调用scipy中的优化算法fmin_bfgs(拟牛顿法Broyden-Fletcher-Goldfarb-Shanno)
- costFunction是自己实现的一个求代价的函数
- initial_theta表示初始化的值
- fprime指定costFunction的梯度
- args是其余测参数，以元组的形式传入，最后会将最小化costFunction的theta返回
```
    result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X,y,initial_lambda))    
```

### 7、运行结果
- data1决策边界和准确度
- ![image](https://github.com/user-attachments/assets/b5555a5e-deff-4b89-9358-4643b66997ed)
- ![image](https://github.com/user-attachments/assets/5f6c3450-6ae8-4b2c-af78-555b65d95a20)
- data2决策边界和准确度
- ![image](https://github.com/user-attachments/assets/d4c174c8-2e8c-41cd-8076-035ee58b9923)
- ![image](https://github.com/user-attachments/assets/4a0c02a8-9efc-4b60-a11f-37eaf0095978)

### 8、使用scikit-learn库中的逻辑回归模型实现
- 导入包
```
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
```
- 划分训练集和测试集
```
    # 划分为训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
```
- 归一化
```
    # 归一化
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
```
- 逻辑回归
```
    #逻辑回归
    model = LogisticRegression()
    model.fit(x_train,y_train)
```
-预测
```
    # 预测
    predict = model.predict(x_test)
    right = sum(predict == y_test)
    
    predict = np.hstack((predict.reshape(-1,1),y_test.reshape(-1,1)))   # 将预测值和真实值放在一块，好观察
    print predict
    print ('测试集准确率：%f%%'%(right*100.0/predict.shape[0]))          #计算在测试集上的准确度
```

## [逻辑回归_手写数字识别_OneVsAll](/LogisticRegression)
- [全部代码]

- ### 1、随机显示100个数字
- 我没有使用scikit-learn中的数据集，像素是20*20px，彩色图如下
![image](https://github.com/user-attachments/assets/0b40b070-b9da-48de-912b-2b8b56f230c2)
灰度图：
![image](https://github.com/user-attachments/assets/5807e8dd-9311-4a6b-ad4d-92ef625ee4fe)
- 实现代码：
```
#显示100个数字
def display_data(imgData):
    sum=0
    ```
    显示100个数（若是一个一个绘制将非常的慢，可以将要画的数字整理好，放在一个矩阵中，显示这个矩阵即可）
    - 初始化一个二维数组
    - 将每行的数据调整成图像的矩阵，放进二维数组
    - 显示即可
    ```
    pad=1
    display_array=-np.ones(pad+10*(20+pad),pad+10*(20+pad)))
    for i in range(10):
        for j in range(10):
             display_array[pad+i*(20+pad):pad+i*(20+pad)+20,pad+j*(20+pad):pad+j*(20+pad)+20] = (imgData[sum,:].reshape(20,20,order="F"))    # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
             sum+=1
    plt.isshow(display_array,camp='gray') #显示灰度图像
    plt.axis('off')
    plt.show()
```
### 2、OneVsAll
- 如何利用逻辑回归解决多分类的问题，OneVsAll就是把当前某一类看成一类，其他所有类别看作一类，这样有成了二分类的问题了
- 如下图，把途中的数据分成三类，先把红色的看成一类，把其他的看作另外一类，进行逻辑回归，然后把蓝色的看成一类，其他的再看成一类，以此类推...
![enter description here][11]
- 可以看出大于2类的情况下，有多少类就要进行多少次的逻辑回归分类
