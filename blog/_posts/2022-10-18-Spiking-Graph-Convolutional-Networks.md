---
layout: post
title: STAM - Spiking Graph Convolutional Networks
truncated_preview: true
excerpt_separator: <!--more--> 
categories:
  - 推荐系统
  - 脉冲神经网络
tags:
  - 脉冲神经网络
  - 图神经网络
  - 推荐系统
  - 论文笔记
---

<div class="message">
图卷积网络和脉冲神经网络的结合
</div>

本文采用的是时间驱动的脉冲时间网络，基于Spikingjelly框架。

<!--more-->

根据对本文及Spiking neural network的了解，将回答几个问题：

- GCN在本文是干嘛的
- 在输入SNN编码器之前卷积的结果是什么？
  -  输出一个卷积得出的矩阵和一个tag

- 为什么说擅长处理时间序列数据。
- 如果擅长处理时序数据，是否意味着可应用于序列推荐。

## GCN在本文是什么作用

GCN是在输入模型之前，进行一个输入至SNN之前初始化的部分，将数据进行图卷积聚合，

（这里需要结合代码）

## GCN聚合之后，通过Encoder转换成spikes,如何转换？

本文提出使用probability-based Bernoulli encoding作为转换node representation到spike signals的方法，本文假设representation的重要性应该和spiking rate成正比的关系，在基于Bernoulli encoder，

> 发射脉冲还有几率的？，难道不是到了膜电位，直接就发射吗？还是说到了膜电位再考虑要不要发射，弄清楚什么是发射率。还有弄清楚什么是脉冲,什么是spike, spike就是脉冲吗？
>
> pre-synaptic spike是不是0.0342之类的连续值电压，还是0，1的离散电压。
>
> 

$\lambda_{i，j}$是i节点的第j个特征的node representation（这是什么意思），这个源自于图卷积的结果。这个值和特征重要性正相关，值越大，被encoder发射spike的几率越大。

把encoding模组视为图的采样过程，将T time steps表示为重复采样的过程，T time同样也能视为信息encoded的分辨率。

（结合代码）

## Charge, fire and reset

这个模组分为fully connected layer层和LIF神经元，connected layer输入是spike，输出是voltages电压至LIF神经元，LIF就能发射脉冲以及重置膜电位。

> 每个节点是特征吗，是整个图是一个文章，还是整个图是所有的数据，每个节点有不同的特征？每个节点是条长向量？
>
> 代码里面的img是什么？encoder之后是什么样的？ 每个节点的每个特征都encoder了嘛？是什么过程？
>
> img就是已经GCN聚合过后的矩阵【bath_size,fearture nums】,encoder了之后，经过possion抽样，每个元素都有不同概率的可以经过抽样，通过了possion的概率的，就为true vise verse.

通常，深度SNN会采用线性或者非线性的组合来处理输入，但是根据SGC来的说法，预测未知的标签，SNN的深度是不重要的，所以本文丢掉了多余的模组，只留下了全连接层。

## Gradient surrogate 

>  梯度替代是在哪里表现的？什么环节发力的，为什么代码中没有？

## 模型可行性分析

提出的方法真的对预测任务有效吗？

相比于其他GCN模型，我们怎么采样过程中控制信息的衰减？

顺便还解释一下spike representation $\psi_{j} o_{j}$有很高的概率接近真实值输出。

文章用了SGC进行对比. SGC和本文采用了相似的框架，将卷积结果H输入全连接层，用真实值的形式。

在SGC中假设真实卷积值是$\lambda$，spike representation是o。

$\operatorname{Pr}\left(o_{j}=1\right)=\lambda_{j}, \quad \operatorname{Pr}\left(o_{j}=0\right)=1-\lambda_{j}$

每个节点的spiking representation **$O_{i, t}^{p r e}=\left(o_{1}, \ldots, o_{j}, \ldots, o_{d}\right)$** 是这样的，节点的每个维度都是独立发射**spike**的。Pr（o=0/1）就是发射脉冲1的几率等于真实值。
