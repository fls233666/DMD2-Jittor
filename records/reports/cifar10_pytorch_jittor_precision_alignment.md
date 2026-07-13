# CIFAR10 PyTorch-Jittor 精度对齐报告

生成日期：2026-07-11

## 结论摘要

| 对齐项 | 当前判断 | 依据 |
| --- | --- | --- |
| CIFAR10 teacher forward | 高精度对齐 | PyTorch/Jittor teacher 输出 MAE 为 `2.2411025e-7`，max AE 为 `9.1642141e-7`。 |
| CIFAR10 训练配置与调度 | 基本对齐 | 两侧均为 CIFAR10、32x32、label_dim=10、batch_size=32、5000 step、GAN 开启、`dfake_gen_update_ratio=5`。 |
| Generator / Guidance loss | 基本对齐 | 5000 step 记录中，核心 loss 的 mean 相对差约 `0.53%` 到 `6.57%`；`summary.json` 中各曲线状态均为 `ok`。 |
| GAN 判别趋势 | 基本一致 | PyTorch / Jittor 的 `gan/real_prob_mean` 都约为 `0.78`，`gan/fake_prob_mean` 都约为 `0.21`。 |
| 生成图质量 | 有本地代理评估，不能等同正式 FID | 已补 PyTorch step5000 图并计算 `Pixel-FID@8x8`；但未完成 50k 样本 Inception-FID。 |

一句话结论：当前 CIFAR10 证据支持“Jittor 侧 DMD2 在 teacher forward、训练调度、DMD/GAN/fake-score loss 尺度和训练趋势上与 PyTorch 侧基本对齐”；但还不能声称完成论文级正式 FID 对齐，也还缺少同随机性的单步反向更新数值对齐。

## 数据来源

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| CIFAR10 teacher forward 对齐 | `teacher-models/cifar10_teacher/forward_alignment_report.json` | 固定输入下比较 PyTorch/Jittor teacher EDM forward 输出。 |
| Jittor CIFAR10 配置 | `DMD2-jittor/configs/cifar10_dmd2.yaml` | 确认 Jittor 侧 CIFAR10 训练设置。 |
| PyTorch CIFAR10 配置 | `DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/config.json` | 确认 PyTorch 侧 CIFAR10 训练设置。 |
| loss 对齐汇总 | `DMD2-records/loss/summary.json` | 汇总 PyTorch/Jittor 5000 step loss 均值、last、count。 |
| Jittor 原始训练日志 | `DMD2-jittor/logs/cifar10_dmd2_5000/train_metrics.jsonl` | Jittor 侧逐 step 训练指标。 |
| PyTorch 原始训练日志 | `DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/train_metrics.jsonl` | PyTorch 侧逐 step 训练指标。 |
| 采样图总览 | `DMD2-records/sample_steps_0_2500_5000/comparison/pytorch_jittor_training_samples_overview.png` | 展示 PyTorch/Jittor 的 step 0/2500/5000 生成效果。 |
| 质量评估报告 | `DMD2-records/generated_image_quality_report.md` | 本地图像质量代理指标与 `Pixel-FID@8x8`。 |
| FID 结果 | `DMD2-records/generated_image_quality_fid/fid_results.json` | 已有 sample grid 拆分后计算的 lightweight FID 结果。 |

## 1. CIFAR10 训练配置对齐

| 配置项 | PyTorch 侧 | Jittor 侧 | 判断 |
| --- | --- | --- | --- |
| dataset | `cifar10` | `cifar10` | 一致 |
| image resolution | `32` | `32` | 一致 |
| label dim | `10` | `10` | 一致 |
| train samples | `50000` | `50000` | 一致 |
| batch size | `32` | `32` | 一致 |
| train steps | `5000` | `5000` | 一致 |
| teacher config | CIFAR10 EDM teacher | CIFAR10 EDM teacher | 一致 |
| generator lr | `2e-4` | `2e-4` | 一致 |
| guidance lr | `2e-4` | `2e-4` | 一致 |
| Adam beta1/beta2 | `0.0 / 0.999` | `0.0 / 0.999` | 一致 |
| warmup step | `0` | `0` | 一致 |
| `dfake_gen_update_ratio` | `5` | `5` | 一致 |
| GAN classifier | enabled | enabled | 一致 |
| generator GAN loss weight | `3e-3` | `3e-3` | 一致 |
| guidance classifier loss weight | `1e-2` | `1e-2` | 一致 |
| diffusion GAN | enabled | enabled | 一致 |
| diffusion GAN max timestep | `1000` | `1000` | 一致 |

补充说明：PyTorch 训练日志 step 范围是 `0..4999`，Jittor 训练日志 step 范围是 `1..5000`。loss 对齐脚本中记录了 `jittor_step_offset=-1.0`，用于把两侧 step 轴对齐。

## 2. CIFAR10 Teacher Forward 数值对齐

| 指标 | PyTorch | Jittor | 差异 |
| --- | ---: | ---: | ---: |
| output mean | `-0.0220896006` | `-0.0220895056` | 非常接近 |
| output std | `0.0422849022` | `0.0422849059` | 非常接近 |
| mean absolute error | - | - | `2.2411025e-7` |
| max absolute error | - | - | `9.1642141e-7` |
| mean relative error | - | - | `5.3543186e-5` |

判断：CIFAR10 teacher checkpoint 转换、EDM precondition wrapper 和基础 forward 路径已经有高精度数值对齐证据。这个结论只覆盖 teacher forward，不等价于完整训练过程逐 step 完全一致。

## 3. 训练调度对齐

| 指标 | PyTorch | Jittor | 判断 |
| --- | ---: | ---: | --- |
| 训练记录数 | `5000` | `5000` | 一致 |
| `compute_generator_gradient` mean | `0.2` | `0.2` | 一致 |
| generator 更新次数 | `1000` | `1000` | 一致 |
| guidance 更新次数 | `5000` | `5000` | 一致 |
| generator 更新节奏 | 每 `5` step 一次 | 每 `5` step 一次 | 一致 |

含义：DMD2 的 TTUR 在当前 CIFAR10 正式记录中已按预期生效。Generator 分支只在 `step % 5 == 0` 时更新；Guidance/fake-score/GAN classifier 分支每个 step 更新。

## 4. CIFAR10 5000 Step Loss 对齐

说明：Generator 相关指标只统计 generator 实际更新的 1000 个 step；Guidance 相关指标统计全部 5000 个 step。GAN 是对抗训练，单点 `last` 波动正常，mean 更适合用于判断整体尺度。

| 指标 | 含义 | count | PyTorch mean | Jittor mean | mean 相对差 | PyTorch last | Jittor last | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `loss_generator` | Generator 总 loss | 1000 | `0.0480564` | `0.0478026` | `0.53%` | `0.0432784` | `0.0444496` | 高度接近 |
| `generator/loss_dm` | DMD 分布匹配主损失 | 1000 | `0.0399357` | `0.0391488` | `1.97%` | `0.0340053` | `0.0360871` | 高度接近 |
| `generator/gen_cls_loss` | Generator 侧 GAN fooling loss | 1000 | `2.7068744` | `2.8846106` | `6.57%` | `3.0910420` | `2.7874856` | 同量级，GAN 波动可接受 |
| `loss_guidance` | Guidance 总 loss | 5000 | `0.5078256` | `0.4919454` | `3.13%` | `0.6598585` | `0.4655226` | 基本对齐 |
| `guidance/loss_fake_mean` | Fake score denoising loss | 5000 | `0.5013468` | `0.4856790` | `3.13%` | `0.6531252` | `0.4605925` | 基本对齐 |
| `guidance/guidance_cls_loss` | Guidance 侧 GAN classifier loss | 5000 | `0.6478751` | `0.6266346` | `3.28%` | `0.6733319` | `0.4930131` | 基本对齐 |

解读：

- 最强证据是 `loss_generator`：两侧 mean 相对差只有 `0.53%`，说明 generator 总优化目标尺度基本一致。
- `generator/loss_dm` 相对差约 `1.97%`，说明 DMD 主损失的符号、归一化和 stop-gradient 路径没有明显错位。
- Guidance 侧总 loss 与 fake-score loss 相对差约 `3.13%`，说明 fake score training 目标整体强度接近。
- `generator/gen_cls_loss` 是 GAN fooling loss，相对差更大但仍同量级；对抗训练中单点和均值波动都比回归类 loss 更明显。

## 5. GAN 判别趋势对齐

| 指标 | PyTorch mean | Jittor mean | PyTorch last | Jittor last | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gan/real_prob_mean` | `0.7809792` | `0.7874056` | `0.7630270` | `0.7938943` | real 图被判为 real 的概率接近 |
| `gan/fake_prob_mean` | `0.2183743` | `0.2121331` | `0.2396039` | `0.1651056` | fake 图被判为 real 的概率接近 |

判断：两侧 GAN classifier 都学到了相同方向的区分能力：real 概率高、fake 概率低。Jittor 最后一个点的 fake prob 更低，但 GAN 单点不稳定，不应作为单独结论。

## 6. 生成图质量与 FID 代理评估

当前已经补充 PyTorch step 5000 图，并整合到最终展示图中：

- PyTorch step 5000：`DMD2-records/sample_steps_0_2500_5000/pytorch/pytorch_step_005000.png`
- 展示总览图：`DMD2-records/sample_steps_0_2500_5000/comparison/pytorch_jittor_training_samples_overview.png`

本地已完成 `Pixel-FID@8x8`。计算方法是把 128x128 sample grid 拆成 16 张 32x32 CIFAR10 小图，再平均池化到 8x8 RGB 像素特征，与 CIFAR10 train 的 50000 张图计算 Frechet Distance。

| 对象 | step | 小图数 | Pixel-FID@8x8 | 说明 |
| --- | ---: | ---: | ---: | --- |
| PyTorch supplemented-grid | 5000 | 16 | `5.7798` | 来自补充的 PyTorch step5000 图 |
| Jittor fixed-grid | 5000 | 16 | `4.4016` | Jittor fixed-noise step5000 |
| Jittor random-grid | 5000 | 16 | `5.4154` | Jittor random-noise step5000 |
| PyTorch local-record | 2500 | 16 | `4.5823` | PyTorch 中期图 |
| Jittor fixed-grid | 2500 | 16 | `4.4840` | Jittor 中期 fixed-noise 图 |

展示用三节点聚合结果：

| 对象 | 小图数 | Pixel-FID@8x8 | 说明 |
| --- | ---: | ---: | --- |
| PyTorch display 0/2500/5000 | 48 | `2.1769` | 混合 step 0/2500/5000，仅用于展示集整体统计 |
| Jittor display 0/2500/5000 | 48 | `2.8948` | 混合 step 0/2500/5000，仅用于展示集整体统计 |

重要边界：

- `Pixel-FID@8x8` 是本地 lightweight FID，不是论文中的正式 Inception-FID。
- 每个最终 grid 只有 16 张小图，样本量太小，只能作为 sanity check。
- PyTorch/Jittor 的 final grid 采样协议不完全等价：Jittor 有 fixed-grid 和 random-grid，PyTorch step5000 是补充图；因此该表适合展示“已有图片的质量检查”，不适合下严格优劣结论。

## 7. 代码语义对齐范围

| 模块 | PyTorch 侧 | Jittor 侧 | 对齐判断 |
| --- | --- | --- | --- |
| 训练入口 | `DMD2-pytorch/scripts/train_cifar10_local_records.sh` / `main/edm/train_edm.py` | `DMD2-jittor/scripts/train_image_dmd2.sh` / `tools/train_image_dmd2.py` | CIFAR10 参数和记录路径可对应 |
| 统一调度 | `main/edm/edm_unified_model.py` | `code/models/unified_model.py` | generator/guidance turn 语义一致 |
| 训练 engine | `main/edm/train_edm.py` | `code/trainer/engine.py` | generator gate + guidance every step 语义一致 |
| DMD loss | `main/edm/edm_guidance.py` | `code/models/guidance.py` / `code/loss/dmd_loss.py` | 公式与 stop-gradient 语义一致 |
| Fake score loss | `main/edm/edm_guidance.py` | `code/models/guidance.py` / `code/loss/regression_loss.py` | fake image detach 与 EDM 权重形式一致 |
| GAN loss | `main/edm/edm_guidance.py` | `code/models/guidance.py` / `code/loss/gan_loss.py` | `softplus(-D(fake))` 与 classifier loss 语义一致 |

## 8. 当前能证明什么

| 证据 | 支持的结论 | 可信度 |
| --- | --- | --- |
| CIFAR10 teacher forward MAE `2.2411025e-7` | teacher 权重转换和 EDM forward 数值实现正确 | 高 |
| `compute_generator_gradient` mean 均为 `0.2` | TTUR 中 generator 每 5 step 更新一次 | 高 |
| `loss_generator` mean 相对差 `0.53%` | Generator 总训练目标尺度基本对齐 | 较高 |
| `generator/loss_dm` mean 相对差 `1.97%` | DMD 主损失实现方向和尺度基本对齐 | 较高 |
| `loss_guidance` mean 相对差 `3.13%` | Guidance/fake-score/classifier 总训练目标基本对齐 | 较高 |
| GAN real/fake prob 均值接近 | GAN classifier 学到的判别趋势一致 | 中到较高 |
| PyTorch/Jittor step 0/2500/5000 展示图与 Pixel-FID@8x8 | 可以做已有图片的质量 sanity check | 中 |

## 9. 目前还不能证明什么

| 未完成项 | 为什么重要 | 当前状态 |
| --- | --- | --- |
| 同 seed、同 batch、同 timestep、同 noise 的单步 loss 对比 | 可证明 generator/guidance forward loss 是否逐数值一致 | 尚未完成 |
| 一次 optimizer step 后参数 delta 对比 | 可证明 backward 和优化器更新方向是否一致 | 尚未完成 |
| GAN classifier bottleneck/logit 逐层数值对比 | 可定位 GAN 判别器内部是否逐层一致 | 尚未完成 |
| 同 checkpoint、同 labels、同 noise 的推理图逐像素对比 | 可证明采样输出是否逐图像接近 | 尚未完成 |
| CIFAR10 正式 Inception-FID / KID / IS | 可证明最终生成质量指标是否严格对齐 | 尚未完成；当前只有 `Pixel-FID@8x8` |

## 10. 汇报建议

| PPT 页 | 标题 | 展示内容 |
| --- | --- | --- |
| 1 | CIFAR10 对齐总览 | 放“结论摘要”和数据来源，先说明本报告只讨论 CIFAR10。 |
| 2 | 训练配置与 TTUR 对齐 | 放训练配置表和 `compute_generator_gradient mean=0.2`。 |
| 3 | Teacher Forward 对齐 | 放 teacher forward mean/std/MAE/max AE。 |
| 4 | Loss 对齐 | 放 5000 step loss 对齐表，重点标出 `loss_generator`、`loss_dm`、`loss_guidance`。 |
| 5 | 生成质量对齐 | 放 step 0/2500/5000 展示图和 `Pixel-FID@8x8` 表。 |
| 6 | 结论与边界 | 放“能证明什么 / 不能证明什么”，主动说明正式 FID 尚未完成。 |

## 最终表述建议

建议在汇报中使用如下表述：

> 在 CIFAR10 分支上，Jittor 侧 DMD2 已经在 teacher forward、训练配置、TTUR 更新节奏、DMD/fake-score/GAN loss 尺度，以及已有采样图的本地质量代理指标上与 PyTorch 侧基本对齐。当前结果可以支撑代码迁移正确性和训练行为一致性的展示，但还不能替代 50k 样本的正式 Inception-FID 评估，也不能证明每个训练 step 在完全相同随机性下逐数值一致。
