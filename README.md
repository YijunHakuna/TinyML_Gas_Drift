# TinyML Gas Sensor Array Drift Compensation

本项目旨在研究如何在资源受限的 STM32 平台上，利用自监督学习（SSL）和端侧自适应技术解决工业传感器的漂移问题。

## 📅 12周计划概览
- **Phase 1**: 问题定义与离线基准（Baseline）构建。
- **Phase 2**: 时序数据自监督表征学习（SimCLR/CPC）。
- **Phase 3**: STM32 模型量化与部署（Int8）。
- **Phase 4**: 端侧自适应（Prototype更新）与漂移监测。

## 🛠️ 技术栈
- **硬件**: STM32F4/H7 (Cortex-M系列)
- **软件**: PyTorch, STM32Cube.AI, TFLite Micro
- **数据集**: UCI Gas Sensor Array Drift Dataset
