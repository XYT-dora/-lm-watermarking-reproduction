# LLM Watermarking - Paper Reproduction

> 复现论文 *"A Watermark for Large Language Models"* (Kirchenbauer et al., ICML 2023 / ICLR 2024)


---

## 📊 实验结果

| 配置 | 有水印 z-score | 无水印 z-score | 困惑度(有水印) | 困惑度(无水印) | 可检测 |
|:---|:---|:---|:---|:---|:---|
| γ=0.25, δ=2.0 | **10.20** | -0.68 | 9.98 | 6.18 | ✅ |
| γ=0.25, δ=5.0 | **20.57** | -0.46 | 15.27 | 6.90 | ✅ |
| γ=0.25, δ=0.5 | 1.72 | 1.70 | 5.47 | 7.55 | ❌ |
| γ=0.5, δ=2.0 | **9.68** | 2.85 | 8.13 | 5.76 | ✅ |

### 关键发现

- **δ 控制水印强度**：δ=2.0 → z≈10.20，δ=5.0 → z≈20.57，δ=0.5 → 无法检测
- **无水印无误报**：所有无水印文本 z < 4.0，假阳性率 0%
- **γ 影响较小**：γ=0.25 比 γ=0.5 水印略强

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```
### 运行实验
```bash
python run_experiments.py
```
-  实验自动运行 4 种配置 × 50 个 prompts = 200 次生成，约需 2-3 小时。

### 查看结果
- 结果保存在\plot目录中

### 📜 License
Licensed under the Apache License, Version 2.0.

### 本实验为论文复现项目，基于官方实现。
