# setup.py
from setuptools import setup, find_packages

setup(
    name="IRAPE",  # 项目/包名称
    version="0.1.0",    # 版本号
    packages=find_packages(), # 自动查找所有包
    python_requires=">=3.10", # Python 版本要求
    install_requires=[
        "torch>=2.0.0",       # 深度学习核心框架
        "transformers>=4.30.0", # Hugging Face 模型库
        "peft",               # 参数高效微调 (Lora等)
        "numpy",              # 矩阵运算
        "tqdm",               # 进度条
        "openai",             # 调用 API (DeepSeek/GPT等)
        "matplotlib",         # 绘图
        "accelerate",         # 建议添加，transformers 训练通常需要
        "sentencepiece",      # 部分分词器(如 Llama/Qwen) 的依赖
    ],
)