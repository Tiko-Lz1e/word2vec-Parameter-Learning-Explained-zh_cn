# word2vec Parameter Learning Explained
Xin Rong ronxin@umich.edu 

## 摘要
Mikolov 等人的文章中的 word2vec 模型和应用在最近两年里吸引了大量的注意。Word2vec 模型所学习的单词的向量表示已经被证明具有语义含义，并且在很多NLP任务中是有用的。当越来越多的研究者开始用 word2vec 或者类似的技术来进行实验时，我发现缺少一份综合性地详细解释单词嵌入模型的参数学习过程，这样会阻止非专业的神经网络研究人员理解这个模型的工作方法。

这篇文章提供了 word2vec 模型的参数更新方程的详细推导和解释，包括原始的连续词袋模型（CBOW）和 skip-gram 模型（SG），以及高级的优化技术，包括分层softmax和负采样（negative sampling）。对于梯度方程的直观解释也会在数学推导旁提供。

在附录中，还提供了一篇关于神经网络和反向传播（backpropagation）基础的综述。我还创建了一个交互性的演示，wevi，以便于直观地理解模型[<sup>1</sup>](#fn1)。

## 1 连续词袋模型

### 1.1 单词汇语境

我们从 [Mikolov 等人的文章（2013a）](#R2)中介绍的最简单的连续词袋模型（CBOW）开始，我们假定在每个语境下只需要考虑一个词，这意味着给定一个上下文单词这个模型将预测一个目标单词，就像一个双词模型（bigram model）一样。建议刚开始了解神经网络的读者在进一步阅读之前，先读一下附录A来大致了解一下重要的概念和术语。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/Figures_1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">插图1：一个语境中只有一个单词的简单CBOW模型</div>
</center>

图示1展示了简化上下文定义下的神经网络。在我们的定义中，词汇量大小为V，隐藏层大小为N。相邻图层上的单元是全连接的。输入是一个一位有效（one-hot）编码的向量，这意味着对于一个给定的上下文单词输入，V单元{x<sub>1</sub>, · · · , x<sub>V</sub> }中将只有一个会是1，其他的会是0。

输入层与输出层之间的权重可以由一个V * N的矩阵W来表示。W的每一行就是是输入层的相关词汇的N维向量表示V<sub>w</sub>。严格来说，W的第i行是V<sub>w</sub><sup>T</sup>。给定一个上下文（一个单词），假定 x<sub>k</sub>=1 且对于 k′≠k 有 x<sub>k′</sub>=0 。我们有
![公式1](./images/formula_1.png)
来将W的第k行复制到h。v<sub>w<sub>I</sub></sub>是输入词汇w<sub>I</sub>的向量表示。这意味着这个隐藏层单元的链接（激活）函数是简单线性的（即直接将输入的加权和传递给下一层）。

从隐藏层到输出层，有一个不一样的N * V大小的权重矩阵W’ = {w<sub>i</sub><sup>′</sup><sub>j</sub>}。使用这些权重值，我们可以为词汇表里的每一个单词计算出一个分数u<sub>j</sub>。
![公式2](./images/formula_2.png)
这里的v<sup>′</sup><sub>Wj</sub>是矩阵W’的第j列。然后我们可以利用softmax，一个对数线性分类模型，来获得单词的后验分布（一种多项分布）。

![公式3](./images/formula_3.png)
这里的y<sub>j</sub>是输出层中第j个单元（the j-the unit）的输出。我们把（1）和（2）代入（3）得到
![公式4](./images/formula_4.png)
注意，这里v<sub>w</sub>和v<sup>'</sup><sub>w</sub>是单词w的两种表示。v<sub>w</sub>来自输入到隐藏层的权重矩阵W的行，v<sup>'</sup><sub>w</sub>来自隐藏层到输出层的矩阵W’的列。在后面的分析中，我们把v<sub>w</sub>叫做单词w的“输入向量”，把v<sup>'</sup><sub>w</sub>叫做单词w的“输出向量”。

#### 隐藏层到输出层的权重更新公式

现在让我们推导出这个模型的权重更新公式。虽然实际的计算是不现实的（下面会进行解释），但洞察这个原始模型也不需要投机取巧。附录A中介绍了反向传播的基础知识。

训练的目标（对于一个训练样本来说）是使公式（4）的值，给定关于权重的输入上下文单词w<sub>I</sub>得到特定的输出单词w<sub>O</sub>（将其在输出层的索引表示为j*）的条件概率，达到最大。

![公式5_6_7](./images/formula_5_6_7.png)
这里的 E = -log p(w<sub>O</sub>|w<sub>I</sub>)是我们的损失函数（我们想要让E最小），并且j*是输出层中某个特定输出单词的索引。注意，这个损失函数可以被理解为两个概率分布之间的交叉熵测量的一种特殊情况。

现在让我们推导出隐藏层和输出层之间的权重更新公式。关于第j个单元的净输入u<sub>j</sub>取E的导数，我们得到
![公式8](./images/formula_8.png)
这里的t<sub>j</sub> = 1 (j = j*)，即，t<sub>j</sub>的值在第j个单元是特定的输出单词时将永远为1，否则t<sub>j</sub> = 0。注意，这个推导只是输出层的预测误差e<sub>j</sub>。

然后我们利用在w<sub>i</sub><sup>'</sup><sub>j</sub>上的导数来得到隐藏层到输出层权重的梯度
![公式9](./images/formula_9.png)

由此，使用随机梯度下降算法，我们得到了隐藏层到输出层权重的权重更新公式
![公式10](./images/formula_10.png)
或
![公式11](./images/formula_11.png)
这里的 η > 0 是学习率，e<sub>j</sub> = y<sub>j</sub> - t<sub>j</sub>，h<sub>i</sub>是隐藏层的第i个单元；v<sup>'</sup><sub>w<sub>j</sub></sub>是w<sub>j</sub>的输出向量。注意，这个更新公式意味着我们需要遍历单词表里的所有可能的单词，检查他们的输出概率y<sub>j</sub>，与他们的期望输出t<sub>j</sub>（0或1）进行比较。如果y<sub>j</sub> > t<sub>j</sub>（“高估”），我们就从v’wj中减去一部分隐藏向量h（即v<sub>w<sub>I</sub></sub>），这样使v<sup>'</sup><sub>w<sub>j</sub></sub>与v<sub>w<sub>I</sub></sub>差距更大；如果y<sub>j</sub> < t<sub>j</sub>（“低估”，仅当t<sub>j</sub> = 1的时候为真，即w<sub>j</sub> = w<sub>O</sub>的时候），我们向v<sup>'</sup><sub>w<sub>O</sub></sub>中加一些h，这样使v<sup>'</sup><sub>w<sub>O</sub></sub>更接近[<sup>3</sup>](#fn3)v<sub>w<sub>I</sub></sub>。如果y<sub>j</sub>与t<sub>j</sub>非常接近，根据更新公式，将会使权重发生非常微小的变化。再次注意，v<sub>w</sub>（输入向量）和v<sup>‘</sup><sub>w</sub>（输出向量）是单词w的两种向量表示

#### 输入层到隐藏层的权重更新公式

得到了W’的更新公式后，我们将目标移向W。我们在隐藏层的输出上取E的导数，得到
![公式12](./images/formula_12.png)
这里h<sub>i</sub>是隐藏层第i个单元的输出；u<sub>j</sub>在公式（2）中定义，是输出层第j个单元的净输入；e<sub>j</sub> = y<sub>j</sub> - t<sub>j</sub> 是输出层第j个单词的预测误差。EH，一个N维向量，是单词表里所有单词输出向量的和，由它们的预测误差加权。

接下来，我们需要将E的导数代入W。首先，回想一下隐藏层对输入层的值进行的线性计算。扩展公式1中的向量符号，我们得到
![公式13](./images/formula_13.png)

现在我们可以在W的每一个元素上取E的导数，得到
![公式14](./images/formula_14.png)

这与x和EH的向量积是相等的，即，
![公式15](./images/formula_15.png)
从这里我们可以得到一个V * N的矩阵。由于x中只有一个元素是非零的，所以（∂E/∂W）中只有一行是非零的，并且这一行的值为一个N维向量EH<sup>T</sup>。我们得到W的更新方程为
![公式16](./images/formula_16.png)
这里的v<sub>W<sub>I</sub></sub>是W中的一行，那个唯一上下文单词的输入向量，也是W中导数不为零的唯一一行。这次迭代后，W中其他的所有行将保持不变，因为他们的导数是零。

直观来讲，由于向量EH是单词表中所有单词由他们的预测误差 e<sub>j</sub> = y<sub>j</sub> - t<sub>j</sub> 加权的输出向量的和，我们可以将公式16理解为向上下文单词的输入向量中添加单词表中每个输出向量的一部分。如果，在输出层中，一个单词w<sub>j</sub>作为输出单词的概率被高估了（y<sub>j</sub> > t<sub>j</sub>），那么上下文单词的输入向量w<sub>I</sub>将趋向于向远离wj的输出向量的方向移动；相反地，如果wj作为输出单词的概率被低估了（y<sub>j</sub> < t<sub>j</sub>），那么输入向量w<sub>I</sub>将趋向于向接近w<sub>j</sub>的输出向量的方向移动；如果w<sub>j</sub>作为输出单词的概率被相当准确地预测，那么他将对w<sub>I</sub>的输入向量的移动产生很小的影响。w<sub>I</sub>的输入向量的移动取决于词汇表里所有向量的预测误差；预测误差越大，一个单词在上下文单词输入向量的移动中施加的影响越显著。

当我们通过遍历从一个训练语料库生成的上下文-目标单词对来迭代更新模型参数时，对向量的影响将不断积累。我们可以想象，一个单词w的输出向量是被w在不同环境下相邻单词的输入向量来回“拖动”的，就像在w和相邻单词的向量之间有真的绳子一样。相似地，一个输入向量也可以看成是被很多输出向量拖动的。这种理解可以让我们想起重力，或力导向布局图。每根假想绳的平衡长度与各个相关词对之间的共同出现强度以及学习率有关。在多次迭代之后，输入向量和输出向量的相对位置将最终稳定下来。

### 1.2 多词汇语境

图示2展示了带有多词汇语境设置的CBOW模型。当我们计算隐藏层输出时，CBOW模型取的是输入上下文单词向量的平均值，并且使用输入层到隐藏层权重矩阵和平均向量的乘积作为输出，而不是直接拷贝输入上下文单词的输入向量。
![公式17_18](./images/formula_17_18.png)
这里的C是上下文中单词的数量，w<sub>1</sub>，···，w<sub>C</sub>是上下文中的单词，v<sub>w</sub>是单词w的输入向量。损失函数为
![公式19_20_21](./images/formula_19_20_21.png)
和公式7，单词汇语境模型中的一样，除了h不一样，是在公式18里而不是公式1里定义的。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/Figures_2.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">插图2：连续词袋模型</div>
</center>

隐藏层到输出层的权重更新公式和单词汇语境中的是一样的（公式11）。我们把它拷贝到这里：
![公式22](./images/formula_22.png)
注意我们需要在每个训练实例的隐藏层到输出层权重矩阵中的每一个元素上应用这个公式。

输入层到隐藏层的的权重更新公式和公式16很相似，除了现在我们需要对上下文中的每一个单词w<sub>I,c</sub>应用下面的等式：
![公式23](./images/formula_23.png)
这里的v<sub>w<sub>I,c</sub></sub>是输入上下文中第c个单词的输入向量；η是一个正学习率；EH = ∂E/∂h<sub>i</sub>是在公式12里给出的。对于这个更新公式的直观理解是和公式16一样的

## 2 Skip-Gram模型

skip-gram模型在[Mikolov等人的文章](#R2)（[2013a](#R2)，[b](#R3)）中被介绍。插图3展示了这种模型。这种模型和CBOW模型是相反的。目标单词在这个模型中处在输入层，而上下文单词在输出层。

我们仍然使用v<sub>w<sub>I</sub></sub>来表示输入层中唯一一个单词的输入向量，因此我们对于隐藏层输出h的定义与公式1中的定义一样，这意味着h就是简单复制了（并且转置了）输入层到隐藏层权重矩阵W中与w<sub>I</sub>有关的一行。我们将h的定义拷贝到下面：
![公式24](./images/formula_24.png)

在输出层中，我们输出C个多项式分布，而不是一个。每一个输出都是使用同一个隐藏层到输出层矩阵计算出来的：
![公式25](./images/formula_25.png)
这里的w<sub>c,j</sub>是输出层第c层面板上的第j和单词；w<sub>O,c</sub>是实际输出上下文单词中的第c个单词；w<sub>I</sub>是唯一的输入单词；y<sub>c,j</sub>是输出层第c层面板上第j个单元的输出。u<sub>c,j</sub>是输出层第c层面板上第j个单元的净输入。因为输出层的每一层面板共享同一组权重，因此
![公式26](./images/formula_26.png)
这里的v<sup>’</sup><sub>w<sub>j</sub></sub>是词汇表中第j个单词w<sub>j</sub>的输出向量，而且也是从隐藏层到输出层权重矩阵W’里某一行里得到的。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/Figures_3.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">插图3：skip-gram模型</div>
</center>

参数更新公式的推导和单词汇语境下的稍有不同。损失函数被替换为
![公式27_28_29](./images/formula_27_28_29.png)
这里j<sub>c</sub><sup>*</sup>是单词表中第c个实际输出上下文单词的索引。

我们对输出层中每层面板上每个单元的净输出u<sub>c,j</sub>取E的导数得到
![公式30](./images/formula_30.png)
和公式8一样。这就是单元的预测误差。为了简化符号，我们定义一个V维向量EI = {EI<sub>1</sub>, ... , EI<sub>V</sub>}作为所有上下文单词的预测误差之和：
![公式31](./images/formula_31.png)

接下来，我们在隐藏层到输出层的矩阵W’上取E的导数，得到：
![公式32](./images/formula_32.png)

因此我们得到了隐藏层到输出层矩阵的更新公式W’，
![公式33](./images/formula_33.png)
或
![公式34](./images/formula_34.png)
对于这个更新公式的直观理解和公式11是一样的，除了预测误差是包含了输出层的所有上下文单词。注意我们需要对所有训练实例中隐藏层到输出层矩阵的每一个元素应用这个更新公式。

对于从输入层到隐藏层矩阵的更新公式的推导与公式12到16相同，除了考虑到预测误差e<sub>j</sub>被替换为EI<sub>j</sub>。我们直接给出更新公式：
![公式35](./images/formula_35.png)
这里的EH是一个N维向量，其中的每个元素定义如下
![公式36](./images/formula_36.png)
对于公式35的直观理解和公式16是一样的。

## 3 优化计算效率

目前为止，我们讨论过的模型（“bigram”模型，CBOW模型，和 skip-gram 模型）都是原始形式，没有使用任何的效率优化技巧。

对于所有的这些模型，词汇表里的每个单词有两种向量表示：输入向量v<sub>w</sub>，输出向量v<sup>’</sup><sub>w</sub>。学习输入向量时不会很费力，但学习输出向量时的代价十分高昂。从更新公式22和23中，我们可以发现，对于每一个训练实例来说，为了更新v<sup>‘</sup><sub>w</sub>，我们需要遍历单词表中的每一个单词w<sub>j</sub>，计算出他们的净输入u<sub>j</sub>，预测概率y<sub>j</sub>（或是skip-gram模型里的y<sub>c,j</sub>），他们的预测误差e<sub>j</sub>（或是skip-gram模型里的EI<sub>j</sub>），最后还要用他们的预测误差来更新他们的输出向量v<sup>’</sup><sub>j</sub>。

对每个训练实例里的所有单词进行这些运算的代价十分昂贵，这导致将我们的算法扩展到大型词汇表或是大型训练语料库是不切实际的。为了解决这个问题，一个直观的想法就是限制每一个训练实例里必须更新的输出向量的数量。实现这一目标的一个优雅的方式是分层softmax；另一个方式是抽样，我们将在下一个章节讨论这种方法。

这两种方法都只优化了更新输出向量时的运算。在我们的推导中，我们关心三个值：（1）E，新的目标函数；（2）∂E/∂v<sup>’</sup><sub>w</sub>，输出向量新的更新公式；（3）∂E/∂h，用于反向传播来更新输入向量的预测误差的加权和

### 3.1 分层softmax

分层softmax是一种高效计算softmax的方法（[Morin and Bengio, 2005](#R5); [Mnih and Hinton, 2009](#R4)）。这个模型使用一个二叉树来表示词汇表里的所有单词。V个单词一定是树的叶子节点。可以证明，树中一定有V-1个内部节点。对于每一个叶子节点，存在唯一的一条从根到叶子节点的路径；这条路径用于估计该叶子节点表示的单词的概率。插图4是一个例子。

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./images/Figures_4.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">插图4：一个分层softmax模型使用的二叉树的例子。白色的节点是单词表里的单词，黑色的节点是内部节点。一条从根节点到w2的示例路径被突出显示。在这个示例中，突出显示的路径长度L(w<sub>2</sub>) = 4。n(w, j)表示从根节点到单词w的路径中的第j个节点。</div>
</center>





## 文章中的注脚
<span id="fn1">1、可以从这里查看这个演示: http://bit.ly/wevi-online.</span>

## 参考文献

<span id="R2">Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013a). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.</span>

<span id="R3">Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.</span>

<span id="R4">Mnih, A. and Hinton, G. E. (2009). A scalable hierarchical distributed language model. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L., editors, Advances in Neural Information Processing Systems 21, pages 1081–1088. Curran Associates, Inc.</span>

<span id="R5">Morin, F. and Bengio, Y. (2005). Hierarchical probabilistic neural network language model. In AISTATS, volume 5, pages 246–252. Citeseer.</span>