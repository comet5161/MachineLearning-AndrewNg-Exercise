# ex4 神经网络学习 作业完成步骤

### 1. 数据可视化
- 随机显示100张手写数字图片

### 2. 完成前向传播
- 给定训练好的神经网络参数
- 计算cost function J
- 计算Regularized的cost function J

### 3. 完成反向传播
- 计算sigmoid函数的导数
- 实现网络参数随机初始化
- 计算代价函数J对每个神经元输入值z的偏导数δ
- 计算所有样本代价函数J对每个神经网络参数θ的偏导数θ_grad，并取平均值（偏置参数θ_0不考虑正则化）
- 使用fmincg函数进行训练

### 4.可视化隐藏层
- 把每个隐藏层的输入参数θ resize成图片显示

### 5.可选练习
- 尝试不同的训练迭代次数 
- 尝试不同的正则化比重λ(lambda )