# word2vec Parameter Learning Explained
Xin Rong ronxin@umich.edu 

## 摘要
Mikolov 等人的文章中的 word2vec 模型和应用在最近两年里吸引了大量的注意。Word2vec 模型所学习的单词的向量表示已经被证明具有语义含义，并且在很多NLP任务中是有用的。当越来越多的研究者开始用 word2vec 或者类似的技术来进行实验时，我发现缺少一份综合性地详细解释单词嵌入模型的参数学习过程，这样会阻止非专业的神经网络研究人员理解这个模型的工作方法。

这篇文章提供了 word2vec 模型的参数更新方程的详细推导和解释，包括原始的连续词袋模型（CBOW）和 skip-gram 模型（SG），以及高级的优化技术，包括分层softmax和负采样（negative sampling）。对于梯度方程的直观解释也会在数学推导旁提供。

在附录中，还提供了一篇关于神经网络和反向传播（backpropagation）基础的综述。我还创建了一个交互性的演示，wevi，以便于直观地理解模型[<sup>1</sup>](#fn1)。

## 连续词袋模型

### 单个词语语境

我们从 [Mikolov 等人的文章（2013a）](#R2)中介绍的最简单的连续词袋模型（CBOW）开始，我们假定在每个语境下只考虑一个词，这意味着给定一个上下文单词这个模型将预测一个目标单词，就像一个双词模型（bigram model）一样。建议刚开始了解神经网络的读者在进一步阅读之前，先读一下附录A来大致了解一下重要的概念和术语。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/Figures_1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">一个语境中只有一个单词的简单CBOW模型</div>
</center>

图示1展示了简化上下文定义下的神经网络。在我们的定义中，词汇量大小为V，隐藏层大小为N。相邻图层上的单元是全连接的。输入是一个一位有效（one-hot）编码的向量，这意味着对于一个给定的上下文单词输入，V单元{x<sub>1</sub>, · · · , x<sub>V</sub> }中将只有一个会是1，其他的会是0。

输入层与输出层之间的权重可以由一个V * N的矩阵W来表示。W的每一行是输入层的相关词汇的N维向量表示V<sub>W</sub>。形式上，W的第i行叫做V<sub>W</sub><sup>T</sup>。给定一个上下文（一个单词），假定 x<sub>k</sub>=1 且对于 k′≠k 有 x<sub>k′</sub>=0 。我们有
![公式1](./images/formula_1.png)
来将W的第k行复制到h。v<sub>wI</sub>是输入词汇W<sub>I</sub>的向量表示。这意味着这个隐藏层单元的链接（激活）函数是简单线性的（即直接将输入的加权和传递给下一层）。

从隐藏层到输出层，有一个不一样的N * V大小的权重矩阵W’ = {w<sub>i</sub><sup>′</sup><sub>j</sub>}。使用这些权重值，我们可以为词汇表里的每一个单词计算出一个分数u<sub>j</sub>。
![公式2](./images/formula_2.png)
这里的v<sup>′</sup><sub>wj</sub>是矩阵W’的第j列。然后我们可以利用softmax，一个对数线性分类模型，来获得单词的后验分布（一种多项分布）。

![公式3](./images/formula_3.png)
这里的y<sub>j</sub>是输出层中第j个单元（the j-the unit）的输出。我们把（1）和（2）代入（3）得到
![公式4](./images/formula_4.png)
这里v<sub>w</sub>和v<sub>w</sub><sup>‘</sup>是单词w的两种表示。v<sub>w</sub>来自输入到隐藏层的权重矩阵W的行，v<sub>w</sub><sup>‘</sup>来自隐藏层到输出层的矩阵W’的列。在后面的分析中，我们把v<sub>w</sub>叫做单词w的“输入向量”，把v<sub>w</sub><sup>‘</sup>叫做单词w的“输出向量”。

#### 更新隐藏层到输出层权重的等式





## 文章中的注脚
<span id="fn1">1、可以从这里查看这个演示: http://bit.ly/wevi-online.</span>

## 参考文献

<span id="R2">Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013a). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.</span>