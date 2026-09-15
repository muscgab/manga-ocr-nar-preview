---
license: cc-by-nc-sa-4.0
library_name: pytorch
pipeline_tag: image-to-text
language:
  - ja
  - en
tags:
  - ocr
  - manga
  - non-autoregressive
  - mps
---

# manga-ocr-nar-preview

[GitHub](https://github.com/muscgab/manga-ocr-nar-preview) · [Hugging Face](https://huggingface.co/muscgab/manga-ocr-nar-preview) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-nar-preview)

**中文**  
`manga-ocr-nar-preview` 是面向日文漫画文字裁剪的非自回归 OCR 预览模型。129 个有序 query 在一次前向中预测最多 128 个正文字符和 EOS。词表覆盖日文与英文。

**English**  
`manga-ocr-nar-preview` is a preview non-autoregressive OCR model for Japanese manga text crops. Its 129 ordered queries predict up to 128 body characters and EOS in one forward pass. The vocabulary covers Japanese and English.

## 使用 / Usage

```bash
git clone https://huggingface.co/muscgab/manga-ocr-nar-preview
cd manga-ocr-nar-preview
pip install -r requirements.txt
python predict.py crop.png --device mps --precision fp32
```

**中文**  
输入应为已经检测并裁剪的文字区域。`--device` 支持 `mps`、`cuda`、`cpu` 和自动选择。MPS 默认使用 FP32 单图优化路径。

**English**  
Inputs should be detected and cropped text regions. `--device` supports `mps`, `cuda`, `cpu`, and automatic selection. MPS uses the optimized FP32 single-image path by default.

## 推理样例 / Inference examples

**中文**  
下列文字裁剪由开放许可字体合成。表中输出来自 step-30000 发布权重的 MPS FP32 推理。

**English**  
The following text crops were rendered with open-licensed fonts. The displayed outputs come from MPS FP32 inference with the step-30000 release weights.

| 输入 / Input | 参考文本 / Reference | 模型输出 / Model output |
|---|---|---|
| <img src="assets/demo/arigatou.png" width="300" alt="ありがとう"> | `ありがとう` | `ありがとう` |
| <img src="assets/demo/dokidoki.png" width="95" alt="ドキドキ"> | `ドキドキ` | `ドキドキ` |
| <img src="assets/demo/kiwotsukete.png" width="230" alt="気をつけて帰ってね"> | `気をつけて帰ってね` | `気をつけて帰ってね` |
| <img src="assets/demo/thank_you.png" width="300" alt="Thank you!"> | `Thank you!` | `Thank you!` |

## Apple MPS：batch=1 / Apple MPS: batch 1

| 项目 / Item | 便携路径 / Portable | 优化路径 / Optimized |
|---|---:|---:|
| P50 前向延迟 / P50 forward latency | 130.43 ms | **83.31 ms** |
| P95 前向延迟 / P95 forward latency | 189.07 ms | **105.89 ms** |
| P50 加速 / P50 speedup | 1.00× | **1.57×** |
| 解码一致 / Decoded parity | — | **64/64** |

**中文**  
测试环境为 Apple M1 Pro、PyTorch 2.10.0、MPS、FP32，CPU fallback 关闭。step-30000 发布权重在 64 张冻结 real-dev 图片上逐张运行。时间包含同步后的模型前向；图片解码、约 2.42 ms 的预处理和 CPU-to-MPS 传输单独计算。优化路径保留宽高比和动态 patch 数，使用位置缓存及 Metal RMSNorm、RoPE、SwiGLU 内核。便携路径和优化路径在全部 64 张图片上产生相同的逐位置 argmax 与解码文本。

**English**  
The benchmark uses an Apple M1 Pro, PyTorch 2.10.0, MPS, and FP32 with CPU fallback disabled. The step-30000 release weights run on 64 frozen real-dev images one at a time. Timings cover synchronized model forward; image decoding, approximately 2.42 ms preprocessing, and CPU-to-MPS transfer are measured separately. The optimized path preserves aspect ratio and dynamic patch count, with position caching and Metal RMSNorm, RoPE, and SwiGLU kernels. The portable and optimized paths produce identical per-position argmax values and decoded text on all 64 images.

**中文**  
先前的 step-10000 测试中，FP16 优化路径达到 74.28 ms P50，64 张中有 63 张与 FP16 基线产生相同文本。FP16 当前保留为实验选项。

**English**  
In an earlier step-10000 test, the optimized FP16 path reached a 74.28 ms P50 and matched the FP16 baseline text on 63 of 64 images. FP16 remains experimental.

## 模型与训练 / Model and training

**中文**  
模型包含 MonkeyOCRv2-B 视觉编码器与 8 层 R2 解码器，共 195,953,570 个参数。解码器宽度为 768，含 12 个注意力头、SwiGLU、Pre-RMSNorm 和 PyTorch SDPA。训练损失由有序交叉熵与权重 0.1 的 occupancy BCE 组成。

**English**  
The model combines a MonkeyOCRv2-B visual encoder with an 8-layer R2 decoder, totaling 195,953,570 parameters. The decoder has width 768, 12 attention heads, SwiGLU, Pre-RMSNorm, and PyTorch SDPA. Training uses ordered cross-entropy plus occupancy BCE with weight 0.1.

**中文**  
训练清单包含 513,361 条筛选后的 AnimeText-OCR 真实伪标签样本和 1,053,440 条合成样本，按 1:1 抽样。合成语料来自日文自然文本、短语组合与少量无语义文本，并加入扫描、JPEG、低分辨率、局部字号变化和轻几何增强。训练使用动态等比输入，面积范围为 224² 至 448²，28 像素对齐。

**English**  
The training inventory contains 513,361 filtered real pseudo-labeled samples from AnimeText-OCR and 1,053,440 synthetic samples, sampled at a 1:1 ratio. Synthetic text uses natural Japanese text, phrase composition, and a small amount of non-semantic text, with scan, JPEG, low-resolution, local font-size, and mild geometric augmentation. Training uses dynamic aspect-ratio-preserving inputs from 224² to 448² area with 28-pixel alignment.

**中文**  
最终训练运行使用 BF16、有效 batch size 256、30,000 个优化器步、AdamW、1000 步 warmup 和余弦衰减。encoder 学习率为 2e-5，decoder 学习率为 3e-4。

**English**  
The final training run uses BF16, an effective batch size of 256, 30,000 optimizer steps, AdamW, 1,000 warmup steps, and cosine decay. The encoder learning rate is 2e-5 and the decoder learning rate is 3e-4.

## 评测 / Evaluation

**中文**  
step-30000 checkpoint 在完整 Manga109-s v2026 的 123,212 个裁剪、87 部作品上达到 75.0308% EM、4.7944% micro-CER 和 99.9911% EOS rate。全部样本均计入分母，参考文本与输出使用固定 NAR 归一化。

**English**  
The step-30000 checkpoint reaches 75.0308% EM, 4.7944% micro-CER, and a 99.9911% EOS rate on all 123,212 crops from 87 works in Manga109-s v2026. Every sample remains in the denominator, and references and outputs use the fixed NAR normalization contract.

## 范围与许可 / Scope and license

**中文**  
当前版本主要面向日文漫画文字裁剪。英文训练量较少，复杂长文本和 65 字以上文本仍有明显改进空间。模型权重采用 CC BY-NC-SA 4.0，代码采用 Apache 2.0。详细来源和归属见 `NOTICE.md`。

**English**  
This release primarily targets Japanese manga text crops. English training exposure is limited, and complex long text and sequences above 64 characters retain substantial room for improvement. Model weights use CC BY-NC-SA 4.0; code uses Apache 2.0. See `NOTICE.md` for provenance and attribution.
