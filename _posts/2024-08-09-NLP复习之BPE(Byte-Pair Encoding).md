---
layout:     post   				    # 使用的布局（不需要改）
title:      NLP复习之BPE 				# 标题 
subtitle:   BPE(Byte-Pair Encoding), #副标题
date:       2024-08-09 				# 时间
author:     Sunstar				# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NLP
    - Word Tokenization
---
# 前言
>前几天面试被问到了关于subword分词的方法，只想起来了BPE，在这里复习一下。

在NLP对语料进行预处理的过程中，需要进行分词、构建词表。其中如果涉及到一些未登录词（即在dev集或test集中出现了但train集中没有出现过的词，Out of Vocabulary-OOV），会对任务性能造成极大影响。为了解决这样的问题，现在的分词器引入了subword。Subword可以是任意的子字符串，或者可以是带有意义的单元，如语素-est或者-er（语素是语言中最小的意义承载单元，unlikeliest这个单词中，就有un-,likely, -est三个语素）。

现代分词方案里，大部分的token是单词，但有些token是频繁出现的语素或者是其他的子词，比如-er。每一个OOV的词都可以用已知的一些subword单元的序列来表示。比如我们有一个语料库:"low, new, newer"，此时"lower"是一个没有见过的单词，在这种情况下，lower可以用"low"和"er"来表示。在有需要的情况下，甚至可以用单个字母的序列来表示。

## BPE算法
基于subword的方法有很多，比如BPE（byte-pair encoding）, WordPiece，unigram language modeling。在这里介绍最简单的BPE。

BPE从词表开始，这个词表是所有单个字符的集合，然后检查训练语料库，选择最频繁相邻的两个符号合并（比如A，B），然后把这个合并符号（AB）加入到词表中，并且把相邻的A，B用AB替换掉。继续计数和合并，直到完成k次，创建了新的k个token（在这里k是算法的一个参数）。生成的词表由原始的字符集和k个新token组成。

BPE算法在通常在单词内部运行（不涉及到跨单词边界合并），因此输入的语料库首先用空格分隔，给出一组字符串，每个字符串对应于一个单词的字符，再加上一个特殊的单词结尾符号"_"，和这些所有字符的计数。示例如下:

low出现5次，lowest出现2次，newer出现6次，wilder出现3次，new出现2次

| Corpus | Vocabulary |
| ------ | ---------- |
| 5 &ensp; l o w _ | _, d, e, i, l, n, o, r, s, t, w |
| 2 &ensp; l o w e s t _ | |
| 6 &ensp; n e w e r _ | |
| 3 &ensp; w i d e r _ | |
| 2 &ensp; n e w _ | |

首先计算所有的相邻符号对，最常见的是"e r"对，一共出现9次，合并er，把er作为一个新token加入到词表

| Corpus | Vocabulary |
| ------ | ---------- |
| 5 &ensp; l o w _ | _, d, e, i, l, n, o, r, s, t, w, er |
| 2 &ensp; l o w e s t _ | |
| 6 &ensp; n e w er _ | |
| 3 &ensp; w i d er _ | |
| 2 &ensp; n e w _ | |

这时候出现最频繁的相邻符号对是"er _"，继续合并

| Corpus | Vocabulary |
| ------ | ---------- |
| 5 &ensp; l o w _ | \_, d, e, i, l, n, o, r, s, t, w, er, er\_ |
| 2 &ensp; l o w e s t _ | |
| 6 &ensp; n e w e r_ | |
| 3 &ensp; w i d e r_ | |
| 2 &ensp; n e w _ | |

接下来是"n e"

| Corpus | Vocabulary |
| ------ | ---------- |
| 5 &ensp; l o w _ | _, d, e, i, l, n, o, r, s, t, w, er, er\_, ne |
| 2 &ensp; l o w e s t _ | |
| 6 &ensp; ne w e r _ | |
| 3 &ensp; w i d e r _ | |
| 2 &ensp; ne w _ | |

继续进行，分别是"ne w", "l o", "lo w", "new er_", "low _"...

注意：在这里测试集中数据的频率不起作用，只有训练数据中的频率会起作用。

### 伪代码
<pre>
function BYTE-PAIR ENCODING(strings C, number of merges k) returns vocab V
    V ← all unique characters in C  <!-- initial set of tokens is characters -->
    for i = 1 to k do  <!-- merge tokens k times -->
        t<sub>L</sub>, t<sub>R</sub> ← Most frequent pair of adjacent tokens in C
        t<sub>NEW</sub> ← t<sub>L</sub> + t<sub>R</sub>  <!-- make new token by concatenating -->
        V ← V + t<sub>NEW</sub>  <!-- update the vocabulary -->
        Replace each occurrence of t<sub>L</sub>, t<sub>R</sub> in C with t<sub>NEW</sub>  <!-- and update the corpus -->
    return V
</pre>

### Python实现
```python
from collections import Counter, defaultdict


def get_vocab(corpus):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("_",)
        vocab[tokens] += freq
    return vocab

def merge_vocab(pair, vocab):
    merge_vocab = {}
    for word in vocab:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merge_vocab[tuple(new_word)] = vocab[word]
    print(merge_vocab)
    return merge_vocab

def get_status(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbles = list(word)
        for i in range(len(symbles) - 1):
            pairs[symbles[i], symbles[i + 1]] += freq

    return pairs

def bpe(corpus, num_merges):
    vocab = get_vocab(corpus)
    for i in range(num_merges):
        pairs = get_status(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        # print(best)
        vocab = merge_vocab(best, vocab)
    return vocab

corpus = {
    "low": 5,
    "lowest": 2,
    "newer": 6,
    "wilder": 3,
    "new": 2
}
num_merges = 15
vocab = bpe(corpus, num_merges)
```