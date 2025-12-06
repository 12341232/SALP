# AMEMA: Adaptive Momentum and EMA-weighted Modeling for Imbalanced Label Distribution Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/wenhuhiji/AMEMA?style=social)](https://github.com/wenhuhiji/AMEMA)


## 项目概述
在分类任务中，**不平衡标签分布**（如长尾数据集）会导致模型偏向多数类、忽视少数类。本项目提出 **AMEMA（Adaptive Momentum + EMA-weighted Modeling）** 方法，通过双模块协同优化，在不增加过多计算成本的前提下，显著提升少数类的识别性能。


## 方法框架
AMEMA 由两个核心模块组成：

### 1. 自适应动量更新模块（Adaptive Momentum）
传统动量更新对所有类别采用相同系数，AMEMA 为每个类别分配动态动量系数：
$$
m_c = m_{base} \cdot \frac{N_{max}}{N_c}
$$
其中：
- $m_{base}$：基础动量系数（默认0.9）
- $N_{max}$：样本最多类别的数量
- $N_c$：第$c$类的样本数量

→ 少数类的动量系数更大，加速其梯度更新。


### 2. EMA加权损失模块（EMA-weighted Loss）
对多数类的损失施加**指数移动平均（EMA）权重**，降低其对模型的主导作用：
$$
\text{Loss}_{EMA}(c) = \text{CE}(y_c, \hat{y}_c) \cdot \gamma^{t}
$$
$$
\text{Loss}_{AMEMA} = \sum_{c=1}^{C} \left( \text{Loss}_{EMA}(c) \cdot m_c \right)
$$
其中：
- $\gamma$：EMA衰减系数（默认0.99）
- $t$：当前训练轮次


## 环境依赖
```bash
# 1. 创建并激活环境
conda create -n amema python=3.8
conda activate amema

# 2. 安装PyTorch（根据CUDA版本调整，示例为CUDA 11.3）
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 3. 安装其他依赖
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1 tqdm==4.64.1


