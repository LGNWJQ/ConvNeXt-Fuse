# ConvNeXt-Fuse

## 说明

- 本代码库的实现为**电子科技大学-信息与通信工程学院-通信工程系-综合课程设计**的一部分
- 这是对DenseFuse的进一步改进：将编码器结构更换为**ConvNeXt**模块

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/77.png)

## 改进效果

* 由于该结构对显存的要求比较高，我们不得已使用更浅的网络结构，更少的编码特征图数量（64->32），最终的版本参数量为25251，使用笔记本的GTX1650(4G)进行训练，使用COCO2014的20000张图训练，batchsize为1。但出乎我们意料的是最终的效果并不逊色于之前经过完全训练的DenseFuse，甚至多种评估方法都处于领先地位，可见该模型还有很大的提升空间

### 与改进前的模型相比较：

#### Tno

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/1.jpg)



![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/2.jpg)



![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/3.jpg)



#### Exposure

![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/4.jpg)



![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/5.jpg)



![](https://raw.githubusercontent.com/LGNWJQ/picgo/main/ConvNeXt-Fuse/6.jpg)

