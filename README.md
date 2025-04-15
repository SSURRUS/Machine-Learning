# Machine-Learning
机器函数学习

# 一、线性回归
 - 全部代码
# 1、代价函数
- J(\theta ) = \frac{1}{{2{\text{m}}}}\sum\limits_{i = 1}^m {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}} <br>
- 其中：{h_\theta }(x) = {\theta _0} + {\theta _1}{x_1} + {\theta _2}{x_2} + ...<br>
- 下面就是要求出theta，使代价最小，即代表我们拟合出来的方程距离真实值最近<br>
- 共有m条数据，其中 {{{({h_\theta }({x^{(i)}}) - {y^{(i)}})}^2}}<br>
- 代表我们要拟合出来的方程到真实值距离的平方，平方的原因是因为可能有负值，正负可能会抵消<br>
- 前面有系数2的原因是下面求梯度是对每个变量求偏导，2可以消去<br>
- 实现代码：<br>
```
# 计算代价函数
def computerCost(X,y,theta):
    m = len(y)
    J = 0
    
    J = (np.transpose(X*theta-y))*(X*theta-y)/(2*m) #计算代价J
    return J
```
-注意这里的X是真实数据前加了一列1，因为有theta(0)<br>
