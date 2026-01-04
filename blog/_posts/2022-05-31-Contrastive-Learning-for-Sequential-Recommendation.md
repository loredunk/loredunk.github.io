---
layout: post
title: Contrastive Learning for Sequential Recommendation
truncated_preview: true
excerpt_separator: <!--more--> 
tags:
  - 对比学习
  - 序列推荐
  - 推荐系统
  - 论文笔记
---

<div class="message">
对比学习之序列推荐
</div>    

<!--more-->

#### contrastive loss function

对比损失函数用于区分两个表示是否源于一个相同用户的历史序列。为了达到这个效果，要最小化同一序列不同增广视角的区别。

对每个用户的序列使用两种随机的增强方法，最后获得2N个被增强的序列，取一个用户的两个增强视角作为正样本对，其他2（N-1）个负样本对。

###  数据增强操作
