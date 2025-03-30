# 医疗文本分类项目

## 项目简介
本项目利用 BERT 模型对医疗文本进行分类。适用于疾病诊断、症状分析等场景。

## 数据集
本项目使用的数据集来源于 [Chinese Disease Question Classification](https://huggingface.co/datasets/whalning/Chinese-disease-question-classification)，该数据集包含中文医疗问题及其分类标签。

## 目录结构
├── logs/                     # 日志文件
├── models/                   # 模型文件
│   └── saved_model/          # 训练好的模型
├── data/                     # 数据文件
│   ├── train.json            # 训练数据
│   └── val.json              # 验证数据
│   └── test.json             # 验证数据
├── train.py                  # 训练脚本
└── README.md                 # 说明文档

## 环境配置
安装依赖：
```bash
pip install -r requirements.txt

python train.py

数据集引用
本项目使用的数据集来自 https://huggingface.co/datasets/whalning/Chinese-disease-question-classification
