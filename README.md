# biyelunwen
我的论文，关于三元组抽取

+ 使用的AllenNLP框架是0.9.1版本，1.0版本的pre-release版本
---
## 数据部分  
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

Q:数据输入时是否需要打乱，生成负样本？
> 可以尝试，但是要看时间来不来的及。比如1：1的采样等等，这里要在datareader里添加比较复杂的逻辑

---
## 模型部分

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

Q:各个loss的权重是否可以调？
> 我这里就很简单的，不去调整， 简单相加好了

中文字向量选择了 https://github.com/liuhuanyong/ChineseEmbedding  
路径在 /storage/gs2018/liangjiaxi/CORPUS/PRETRAINED/ChineseEmbedding/model/token_vec_300.bin

预训练模型有哪些？  

Q: BERT的未登录词怎么办？
> 使用词表替换。get_word_from_pretrained函数实现这个功能

```
020-02-14 18:08:31,278 - my_library.myutils - INFO - 瘿 不在词表，但是可以修改[unused2]->2 变成瘿->2 9711
2020-02-14 18:08:31,300 - my_library.myutils - INFO - — 不在词表，但是可以修改[unused3]->3 变成—->3 9710
2020-02-14 18:08:31,301 - my_library.myutils - INFO - S 不在词表，但是可以修改[unused4]->4 变成S->4 9709
2020-02-14 18:08:31,301 - my_library.myutils - INFO - M 不在词表，但是可以修改[unused5]->5 变成M->5 9708
2020-02-14 18:08:31,301 - my_library.myutils - INFO - C 不在词表，但是可以修改[unused6]->6 变成C->6 9707
2020-02-14 18:08:31,305 - my_library.myutils - INFO - 蝽 不在词表，但是可以修改[unused7]->7 变成蝽->7 9706
2020-02-14 18:08:31,317 - my_library.myutils - INFO - “ 不在词表，但是可以修改[unused8]->8 变成“->8 9705
2020-02-14 18:08:31,317 - my_library.myutils - INFO - ” 不在词表，但是可以修改[unused9]->9 变成”->9 9704
2020-02-14 18:08:31,319 - my_library.myutils - INFO - K 不在词表，但是可以修改[unused10]->10 变成K->10 9703
2020-02-14 18:08:31,322 - my_library.myutils - INFO - 鵙 不在词表，但是可以修改[unused11]->11 变成鵙->11 9702
2020-02-14 18:08:31,343 - my_library.myutils - INFO - 　 不在词表，但是可以修改[unused12]->12 变成　->12 9701
2020-02-14 18:08:31,350 - my_library.myutils - INFO - B 不在词表，但是可以修改[unused13]->13 变成B->13 9700
2020-02-14 18:08:31,353 - my_library.myutils - INFO - 瓘 不在词表，但是可以修改[unused14]->14 变成瓘->14 9699

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


# 实验记录
## 实验一
### 二阶段模型
#### 不同神经网络的表现
普通的模型要训练200epochs，patience设置为50。为了训练更为充分  
预训练语言模型则设置为25epochs，无patience

| 模型名称| 验证集| 测试集 |
| ----| ----| ---- |
|r lstm 2| 35.2 | .9 
|w_lstm_2| 60.9| 28.5
