---
layout:     post   				    # 使用的布局（不需要改）
title:      NLP复习之HMM与Viterbi算法				# 标题 
subtitle:   HMM, Viterbi #副标题
date:       2024-08-09 				# 时间
author:     Sunstar				# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NLP
    - DP
---

# 前言
>同样的，面试被问到了Viterbi算法，索性一起复习一下HMM和Viterbi

HMM（Hidden Markov Model）是一个序列标记算法，可以应用到词性标注（Part-of-Speech Tagging），语音里应该也有使用到。

## HMM
HMM是一种概率序列模型，给定一个单元序列(words, letters, morphemes, sentences, whatever)，计算可能的标签序列并且选择最佳标签序列

### 马尔科夫链(Markov Chains)
HMM基于增强马尔科夫链。**马尔可夫链**（Markov chain）是一种模型，用来描述一系列随机变量（或者说状态）的序列，这些随机变量可以从一个集合中取值。马尔可夫链告诉我们这些序列中各个状态的概率。这些集合可以是代表任何事物的单词、标签或者符号。

马尔科夫链假定：如果我们想要预测序列中的未来，最重要的是当前的状态。当前状态之前的所有状态除了通过当前状态之外，对未来没有任何影响。

更正式地说，考虑一个状态变量序列，$q_{1}, q_{2}, \cdots, q_{i}$,一个马尔科夫模型体现了对这个序列的概率的**马尔科夫假设**：当预测未来的时候，过去不重要，只有现在重要(When predicting the future, the past doesn't matter, only the present)
$$Markov~Assumption: P(q_{i} = a |q_{1}\cdots q_{i - 1}) = P(q_{i}=a|q_{i - 1})$$

形式上，马尔科夫链由以下部分组成：

| 符号 | 描述 |
|:----:|:----:|
| $Q=q_{1}q_{2}\cdots q_{N}$ | N个状态的集合|
| $A=a_{11}a_{12}\cdots a_{N1}\cdots a_{NN}$| 转移概率矩阵$A$，每一个$a_{ij}$代表从状态$i$到$j$的概率，且满足$\forall i$, $\sum_{j=1}^{n}a_{ij}=1$|
|$\pi = \pi_{1}, \pi_{2}, \cdots, \pi_{N}$| 状态的初始概率分布，$\pi_{i}$是马尔科夫链从状态$i$开始的概率。某些状态$j$可能有$\pi_{j} = 0$， 表示它们不能是初始状态。并且$\sum_{i=1}^{n}\pi_{i} = 1$|

### 隐马尔可夫模型(The Hidden Markov Model)
在计算一系列可观察到的事件的时候，马尔科夫链非常有用，但是在很多情况下，我们感兴趣的事情是隐藏的，我们无法直接观察到这些内容。比如我们通常无法观察文本中词性标签。相反，我们看到单词，并且从单词序列中推断标签，这些标签就是隐藏的。

隐马尔可夫模型（HMM）使我们能够描述和分析两个层次的事件：可观测事件（比如我们在输入中看到的单词）和隐藏事件（比如词性标注），这些隐藏事件是我们认为的概率模型中的因果因素。一个HMM模型由以下几部分组成：

| 符号 | 描述 |
|:----:|:----:|
| $Q=q_{1}q_{2}\cdots q_{N}$ | N个状态的集合|
| $A=a_{11}a_{12}\cdots a_{N1}\cdots a_{NN}$| 转移概率矩阵$A$，每个$a_{ij}$代表从状态$i$到$j$的概率，且满足$\forall i$, $\sum_{j=1}^{n}a_{ij}=1$|
| $B = b_{i}(o_{t})$| 观察的可能性的一个序列，也被称为发射概率(emission probability)，每个都表示从状态$q_{i}$生成观察值$o_{t}$($o_{t}$从词表中提取)的概率，|
|$\pi = \pi_{1}, \pi_{2}, \cdots, \pi_{N}$| 状态的初始概率分布，$\pi_{i}$是马尔科夫链从状态$i$开始的概率。某些状态$j$可能有$\pi_{j} = 0$， 表示它们不能是初始状态。并且$\sum_{i=1}^{n}\pi_{i} = 1$|

提供一个输入$O=o_{1}o_{2}\cdots o_{T}$:一系列的T个观测值，每个都从词表V中提取。

一阶隐马尔科夫模型实例化两个简化的假设。首先，和一阶马尔科夫链一样，特定状态的概率仅取决于先前的状态：
$$Markov ~ Assumption: P(q_{i}|q_{1}, \cdots, q_{i - 1}) = P(q_{i}|q_{i - 1})$$
其次，输出观测值$o_{i}$的概率仅取决于产生观测值$q_{i}$的状态，不取决于任何其他的状态或者其他的观测值：
$$Output~Independence: P(o_{i}|q_{1}, \cdots, q_{i}, \cdots, q_{T}, o_{1}, \cdots, o_{i}, \cdots, o_{T}) = P(o_{i}|q_{i})$$

#### HMM标记器的组成
一个HMM包含两个组成成分，即$A$，$B$概率。

$A$矩阵包含标签的转移概率$P(t_{i}|t_{i - 1})$，表示给定前一个标签的情况下，出现当前标签的概率。比如，像will这样的情态动词后面很可能接一个动词的基本形式。通过计算在标记语料库里看到第一个标签的次数，计算第一个标签后面跟随第二个标签的概率，来计算这个转移概率的最大似然估计：
$$P(t_{i}|t_{i - 1}) = \frac{C(t_{i - 1}, t_{i})}{C(t_{i - 1})}$$

Emission Probabilities $B$, $P(w_{i}|t_{i})$表示给定标签的情况下，与给定单词相关联的概率，发射概率的最大似然估计的计算公式为：
$$P(w_{i}|t_{i}) = \frac{C(t_{i}, w_{i})}{C(t_{i})}$$

B可以这样解释：如果我们要生成一个$t_{i}$(可能是情态动词)，这个情态动词是$w_{i}$（可能是will）的可能性有多大

#### 用HMM标记做解码
对于任何一个包含隐藏变量的模型，确定与观测序列相对应的隐藏状态序列的任务就是解码。即

解码: 给定输入HMM $\lambda = (A, B)$和一个观测序列$O=o_{1},\cdots,o_{2},\cdots,o_{T}$,找到最可能的状态序列$Q=q_{1}q_{2}q_{3}\cdots q_{T}$.

在词性标注(part-of-speech tagging)任务里，HMM解码的目标就是给定$n$个单词的观测序列$w_{1}\cdots w_{n}$，选择最有可能的标签序列$t_{1},\cdots,t_{n}$
$$\hat t_{1:n} = \argmax_{t_{1}\cdots t_{n}} P(t_{1}\cdots t_{n}|w_{1}\cdots w_{n})$$

在HMM中，使用Bayes' rule来计算（
$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$
），
$$\hat t_{1:n} = \argmax_{t_{1}\cdots t_{n}} \frac{P(w_{1}\cdots w_{n}|t_{1}\cdots t_{n})P(t_{1}\cdots t_{n})}{P(w_{1}\cdots w_{n})}$$

可以通过去掉分母$P(w_{1}^{n})$来简化公式：

$$\hat t_{1:n} = \argmax_{t_{1}\cdots t_{n}} P(w_{1}\cdots w_{n}|t_{1}\cdots t_{n})P(t_{1}\cdots t_{n})$$

HMM标注器做了两个进一步的简化的假设，第一个是单词出现的概率只取决于它自己的标签，与临近的单词和标签无关：
$$P(w_{1}\cdots w_{n}|t_{1}\cdots t_{n}) \approx \prod_{i=1}^{n}P(w_{i}|t_{i})$$

第二个假设就是bigram假设，一个标签的概率只依赖于前一个标签，而不是整个标签序列：
$$P(t_{1}\cdots t_{n}) \approx \prod_{i=1}^{n}P(t_{i}|t_{i - 1}) $$

把这两个公式带入到上面的公式中，得到：
$$\hat t_{1:n} = \argmax_{t_{1}\cdots t_{n}} P(w_{1}\cdots w_{n}|t_{1}\cdots t_{n})P(t_{1}\cdots t_{n}) \approx \argmax_{t_{1}\cdots t_{n}} \prod_{i=1}^{n} \overbrace{P(w_{i}|t_{i})}^{emission} \overbrace{P(t_{i}|t_{i - 1})}^{transition}$$

这里的内容和$B$（发射概率），$A$（转移概率）完全对应。


#### 维特比算法(Viterbi Algorithm)
维特比算法用作HMM的解码算法，是一个动态规划算法：

