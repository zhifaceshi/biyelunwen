# biyelunwen
我的论文，关于三元组抽取

+ 使用的AllenNLP框架是0.9.1版本，1.0版本的pre-release版本
---
## 数据部分  
_train 等文件是短文本 8W
train 则是全量的文件 13W 
> 思考了是否需要用全量的数据，后来想了想是不用的。原因在于，本文研究的主要内容并不是数据量的问题，而是模型的问题。
> 我们应当聚焦于模型，而不是数据。

Q: 在数据处理部分，一句话出现多个subject，应该全部标注出，还是部分标注出？
> + 仅标注出第一次出现的位置优势在于简单 
> + 如果全标注出，则在解码时可能会重复解码，重复预测两次张三
例如张三的父亲是李四，张三的母亲是王五。重复解出两次张三，增加了冗余度，但是可以取交并集合来提升准确度。（这个可以单独研究？）

区分训练集和验证集。单独写两个数据读取类，用于读取数据。  
Q: pretrained的词表如何映射到？  
> PretrainedTransformerIndexer类的  
>_add_encoding_to_vocabulary_if_needed方法自动给词表进行添加了

Q:数据输入时是否需要打乱，生成负样本？
> 可以尝试，但是要看时间来不来的及。比如1：1的采样等等，这里要在datareader里添加比较复杂的逻辑  
> 但是后来看，似乎不打乱效果更好？
---
## 模型部分

+ 不同的结构
+ 不同的层数
+ 不同的神经网络

**重点在于：我可以做出什么有创新的贡献？**

### 细节 对span的处理。

batched_index_select,先将index转成1维，然后再选取。比for循环要快~
batched_span_select   

### **自问自答**
Q: span范围后，取得subject如何取？
> 范围相加、范围边界、范围求均、范围LSTM、返回自注意力
SpanExtractor 这个类是专门处理以上的逻辑


Q:如何将这个向量和embedding结合
> 重复后相加，重复后线性变换再相加，重复后取平均  
> 相加相当于是权重为1， 变换相当于是全连接， mix 相当于是只有两个参数

Q:各个loss的权重是否可以调？
> 我这里就很简单的，不去调整， 简单相加好了。调整又可以写一篇论文。

中文字向量选择了 https://github.com/liuhuanyong/ChineseEmbedding  
路径在 /storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/ChineseEmbedding/model/token_vec_300.bin


Q: BERT的未登录词怎么办？
> 使用词表替换。cibiao_sub.py实现这个功能


Q: 事实上所有的模型都会有一个问题。即匹配的问题
> 每个阶段的模型都有自己的匹配方法，一般是指最近匹配

---
# 预期贡献
+ 关系抽取思路的较为完整的理论分析，从理论上分析了传统 `命名实体识别 + 关系分类` 思路的不足。
+ 横向对比了三个大框架（一阶段模型、二阶段模型、三阶段模型）的优缺点，实际手动实现，并记录了可能存在的问题。
+ 针对一阶段模型，提出自己（原创？存疑，目前无法google，回校搜索尝试）的 encoder + matrix attention模型.  
    + 解码端的尝试如下：
        + 线性attention （W*[E1 opt E2]） 这里的opt可以包含 + - * / concat 等操作。
        + 参考`esim` 模型 [E1; E2; E1-E2; E1*E2] 组合，查看是否会有效果提升
        + 双线性attention （E1 * W * E2）
    + 编码端的尝试不是重点
        + Bi-LSTM 2层 （作为基准）
        + BERT模型 6层 在找到合适的解码端之后，尝试与此结合会不会有效果上的提升
    + 预期结论： 
        + 信息抽取领域 xxxx 解码 效果最好
        + BERT模型 + 此解码 相比 效果有提升
    
    
+ 针对二阶段模型，涉及`encoder + span extractor + decoder` 三个模块
    + 编码端尝试
        + random embedding 300维度 + BiLSTM 2层的编码能力
        + word2vec embedding 300维度 + BiLSTM 2层的编码能力
        + random embedding 768维度 + BiLSTM 2层的编码能力
        + BERT 2 层 的编码能力
        + BERT 6 层 的编码能力
    + span extractor尝试
        + endpoint 取范围两端融合方式 (相加、相减等)
        + 自注意力加权的方式 sum(sigmoid(W * E) * E)
    + 解码端尝试
        + FCN解码
        + CNN解码
        + LSTM解码
    
    + 预期结论：
        + word2vec 的引入 （是否） 增加了模型的表达能力
        + BERT模型的编码 （是否） 相比传统模型有提高
        + BERT模型的层数加深 （是否） 效果会更好
        + (endpoint, 自注意加权) 抽取效果会更好
        + FCN 多层+tanh 相比简单单层 效果（是否） 会有提升
        + 卷积核为3的卷积层 相比一层 FCN 效果（是否） 会有提升
        + 一层LSTM 相比 一层FCN 效果（是否） 会有提升

+ 针对三阶段模型， 仍然涉及`encoder + span extractor + decoder`
    + 针对二阶段选择的比较好的模块，研究三阶段效果
    + 针对分类问题有以下方法：
        + sum方法
        + 取最后方法
        + cnn + pool
        + lstm

# 实验记录
## 实验一
### 二阶段模型
#### 不同神经网络的表现
普通的模型要训练100epochs，patience设置为25。为了训练更为充分  
预训练语言模型则设置为25epochs，无patience




| 模型名称| 验证集p| 验证集r | 验证集f1 |测试集 p|测试集 r|测试集 f1| 
| ----| ----| ---- |  ---- | ---- | ---- | ---- |
|r768 bilstm 2 | 81.98| 55.81| 66.41 | 82.15 | 56.92 | 67.25|
r300 lstm_2_fcn_1 | 
w_lstm_2_fcn_1_300|
bert_2_fcn_1|


## 实验二
### 三阶段模型
设置与上相同，观察现象
遇到的问题是，训练速度慢、loss降低很慢。如何解决？
因为模型结构设计的比较复杂


## 实验三
### 一阶段模型
问题的关键在于是对指向进行建模。  
即，如果抽取出同个关系下，多个主体和客体，如何匹配的问题。
> 突然想起了之前的encoder + matrix attention。因为attention矩阵是一个有向图，故可以进行指向。
> 所以必须是双向的，而非单向

+ 训练了3个epoch，模型的loss下降到0.0036，但是解码的 值为 `0`。
    + 尝试进行debug。
    + 怀疑是，全预测为 0，假设10 * 10，只有一个位置， 则 就算全预测0， 平均下的loss也是0.01
        + 解决方案是 添加mask将padding屏蔽掉 + new_mask的权重调整，让




预训练模型有哪些？  
```
distillbert中文是繁体的？
vocab.get('眼', '？？')
   Out[8]: '？？'
   vocab.get('爱', '？？')
   Out[9]: '？？'
   vocab.get('梦', '？？')
   Out[10]: '？？'
   vocab.get('想', '？？')
   Out[11]: '？？'
   vocab.get('疯', '？？')
   Out[12]: '？？'
   vocab.keys()
   Out[13]: odict_keys(['[PAD]', '[

roberta 
{"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, ".": 4, "Ġthe": 5, ",": 6, "Ġto": 7, "Ġand": 8, "Ġof": 9, "Ġa": 10, "Ġin": 11, "-": 12, "Ġfor": 13, "Ġthat": 14, "Ġon": 15, "Ġis": 16, "âĢ": 17, "'s": 18, "Ġwith": 19, "ĠThe": 20, "Ġwas": 21, "Ġ\"": 22, "Ġat": 23, "Ġit": 24, "Ġas": 25, "Ġsaid": 26, "Ļ": 27, "Ġbe": 28, "s": 29, "Ġby": 30, "Ġfrom": 31, "Ġare": 32, "Ġhave": 33, "Ġhas": 34, ":": 35, "Ġ(": 36, "Ġhe": 37, "ĠI": 38, "Ġhis": 39, "Ġwill": 40, "Ġan": 41, "Ġthis": 42, ")": 43, "ĠâĢ": 44, "Ġnot": 45, "Ŀ": 46, "Ġyou": 47, "ľ": 48, "Ġtheir": 49, "Ġor": 50, "Ġthey": 51, "Ġwe": 52, "Ġbut": 53, "Ġwho": 54, "Ġmore": 55, "Ġhad": 56, "Ġbeen": 57, "Ġwere": 58, "Ġabout": 59, ",\"": 60, "Ġwhich": 61, "Ġup": 62, "Ġits": 63, "Ġcan": 64, "Ġone": 65, "Ġout": 66, "Ġalso": 67, "Ġ$": 68, "Ġher": 69, "Ġall": 70, "Ġafter": 71, ".\"": 72, "/": 73, "Ġwould": 74, "'t": 75, "Ġyear": 76, "Ġwhen": 77, "Ġfirst": 78,"Ġshe": 79, "Ġtwo": 80, "Ġover": 81, "Ġpeople": 82, "ĠA": 83, "Ġou
```

# 经验之谈
+ simple is beauty
+ 解码过程要花很多时间

# 参考
[西多士NLP 全面的总结了各类抽取模型](https://www.cnblogs.com/sandwichnlp/p/12049829.html)  
[徐阿衡 一篇类似于综述性质的文章](http://www.shuang0420.com/2018/09/15/%E7%9F%A5%E8%AF%86%E6%8A%BD%E5%8F%96-%E5%AE%9E%E4%BD%93%E5%8F%8A%E5%85%B3%E7%B3%BB%E6%8A%BD%E5%8F%96/)


Dudek D . Automated Information Extraction and Classification of Matrix-Based Questionnaire Data[C]// International Conference on Systems Science. Springer International Publishing, 2017.