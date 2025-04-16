# Machine-Learning
机器函数学习

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
- ![image](https://github.com/user-attachmen![image](https://github.com/user-attachments/assets/a9c5e024-354c-4caf-b897-d74256fb1814)
ts/assets/f4f624d6-c039-4115-8bc3-5facfb64fc44)

### 2、梯度
- 同样对代价函数求偏导：
- \frac{{\partial J(\theta )}}{{\partial {\theta j}}} = \frac{1}{m}\sum\limits{i = 1}^m {[({h_\theta }({x^{(i)}}) - {y^{(i)}})x_j^{(i)}]}
- 可以看出来与线性回归的偏导数一致
- 推导过程
- ![image](https://github.com/user-attachments/assets/57b1549c-12fb-48f7-aa87-407ac43cfdec)

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
