# biyelunwen
我的论文，关于三元组抽取

+ 使用的AllenNLP框架是0.9.1版本，1.0版本的pre-release版本
---
##数据部分  
_train 等文件是短文本 7W
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


---
##模型部分

+ 不同的结构
+ 不同的层数
+ 不同的神经网络

重点在于：我可以做出什么有创新的贡献？

### 细节1 对span的处理。

batched_index_select,先将index转成1维，然后再选取。比for循环要快~
batched_span_select   
Q: span范围后，取得subject如何取？
> 范围相加、范围边界、范围求均、范围LSTM、返回自注意力
SpanExtractor 这个类是专门处理以上的逻辑


Q:如何将这个向量和embedding结合
> 重复后相加，重复后线性变换再相加，重复后取平均  
> 相加相当于是权重为1， 变换相当于是全连接， mix 相当于是只有两个参数


中文字向量选择了 https://github.com/liuhuanyong/ChineseEmbedding  
路径在 /storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/ChineseEmbedding/model/token_vec_300.bin

预训练模型有哪些？  

|  模型名   | 是否支持中文？  |
|  ----  | ----  |
| bert-base-chinese  | 是 |
| albert-base-v2  | 是 |
| distilbert-base-uncased  | 是 |
| gpt2  | 否 |
| roberta-base  |  |