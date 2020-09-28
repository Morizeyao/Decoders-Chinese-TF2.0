# 基于Tensorflow 2.0 与 Huggingface 的 [Transformers](https://github.com/huggingface/transformers) 的 GPT2 与 Transformer-XL 训练代码
# （本项目不保证能稳定运行）

## 使用方法

- pip install -r requirements.txt
- 将scripts文件夹中的几个sh文件拷贝到拷贝到根目录。
- 运行prepare_data.sh预处理。
- 然后运行train_gpt2_keras.sh训练。
- 运行generate_gpt2_keras.sh生成。

## 数据格式

- 本项目采用的语料格式为一个txt文件，每一行是一个json的列表，这个列表中包含了一篇文章。
