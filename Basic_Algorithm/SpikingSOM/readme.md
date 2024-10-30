# SpikingSOM - 脉冲定时依赖可塑性自组织映射学习算法
本工具可在MNIST，CIFAR-10 以及 STL-10 三个数据集上进行脉冲自组织映射网络的训练

MNIST 的编码器使用简单4层CNN，CNN 预训练通过运行 CNNTrain.py 完成，第13行 mode='SSL' 为自监督学习（默认），将mode设为'SL'则为监督学习
CIFAR-10 和 STL-10 的编码器使用 ResNet18 架构，ResNet18的预训练参数需要使用其他工具包完成，在本压缩包内提供了训练完成的模型权重

## 训练 Train
- 在 MNIST 上训练SOM：运行SOMTrain_MNIST.py
- 在 CIFAR-10 或 STL-10 上训练SOM：运行SOMTrain_Resnet.py ；第150行切换测试集
每运行2000个样本会在 Results/images/{Dataset} 生成编号为 epoch+iteration+label 的一组热点图，label={0, ... , 9} 为图像标签。热点图表示了激活神经元的位点，理想状态下，每个标签的激活区域应该不同。

若要查看U矩阵，将SOM U Matrix第28行的路径更替为要测试的权重，并运行。理想状态下，U矩阵上应当出现大于等于10个浅色区块
* 重复运行 SOMTrain 时会将原参数覆盖，在查看U矩阵时请注意