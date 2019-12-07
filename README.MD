# 基于Tensorflow 2.0 与 Huggingface 的 Transformers 的 GPT2 与 Transformer-XL 训练代码

- 目前只有GPT2部分可用。
- 即使是GPT2部分也仅在CPU机器上测试通过。

## 使用方法

- pip install -r requirements.txt
- 将scripts文件夹中的三个sh文件拷贝到拷贝到根目录。
- 运行prepare_data.sh预处理。
- 然后运行train_gpt2.sh训练。
- 运行generate.sh生成。