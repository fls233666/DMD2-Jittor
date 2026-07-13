# PyTorch-Jittor 精度对齐情况汇总

生成日期：2026-07-10

## 总体结论

| 结论项 | 当前判断 | 说明 |
| --- | --- | --- |
| forward 数值对齐 | 已有强证据 | CIFAR10 teacher forward 和 ImageNet64 generator forward 都有逐 tensor 误差统计，误差在较小范围内。 |
| CIFAR10 训练 loss 对齐 | 基本对齐 | 5000 step 训练记录中，核心 loss 的均值尺度接近，`summary.json` 中对应曲线状态均为 `ok`。 |
| 训练调度与 loss 公式 | 代码路径对齐 | TTUR、DMD loss、fake score loss、GAN generator loss、GAN classifier loss 的代码语义与 PyTorch 侧一致。 |
| 最终生成质量/FID 对齐 | 暂无正式结论 | 当前记录中未找到完整 FID/IS 或同 checkpoint、同 noise 的样本质量对齐结果。不能把 loss 对齐直接等同于论文指标复现。 |

## 数据来源

| 类型 | 文件/路径 | 用途 |
| --- | --- | --- |
| loss 对齐汇总 | `/home/koishi/DMD2/DMD2-records/loss/summary.json` | 汇总 PyTorch 与 Jittor CIFAR10 5000 step 训练 loss 的 count、mean、last、min、max。 |
| PyTorch 训练配置 | `/home/koishi/DMD2/DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/config.json` | 确认 CIFAR10、batch size 32、train iters 5000、GAN 开启、`dfake_gen_update_ratio=5`。 |
| PyTorch 训练统计 | `/home/koishi/DMD2/DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/summaries/train_metrics_summary.json` | 确认 PyTorch 侧 generator 每 5 step 更新一次，guidance 每 step 更新一次。 |
| 对齐报告 | `/home/koishi/DMD2/progress-docs/alignment_report.md` | 记录 forward/checkpoint 对齐、TTUR 与 loss 公式映射、剩余缺口。 |
| Jittor 训练日志 | `/home/koishi/DMD2/DMD2-jittor/logs/cifar10_dmd2_5000/train_metrics.jsonl` | Jittor 侧 CIFAR10 5000 step 原始训练记录。 |
| PyTorch 训练日志 | `/home/koishi/DMD2/DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/train_metrics.jsonl` | PyTorch 侧 CIFAR10 5000 step 原始训练记录。 |

## 1. Forward / Checkpoint 数值对齐

| 对齐项 | PyTorch 侧 | Jittor 侧 | 差异/误差 | 判断 | 来源 |
| --- | --- | --- | --- | --- | --- |
| ImageNet64 generator checkpoint 转换 | 输入 key 数 `553` | 转换后 key 数 `541` | 丢弃确定性的 `resample_filter` key `12` 个；unexpected/missing/shape mismatch/duplicate 均为 `0` | 权重映射通过 | `progress-docs/alignment_report.md` |
| ImageNet64 generator forward | output mean/std = `0.15208736 / 0.65460753` | output mean/std = `0.15208854 / 0.65460896` | MAE `5.81759195e-6`；median AE `2.71201134e-6`；max AE `1.75967813e-4`；RMSE `1.26740533e-5` | forward 数值基本对齐 | `progress-docs/alignment_report.md` |
| CIFAR10 teacher EDM forward | output mean/std = `-0.0220896006 / 0.0422849022` | output mean/std = `-0.0220895056 / 0.0422849059` | MAE `2.2411025e-7`；max AE `9.1642141e-7` | teacher forward 高精度对齐 | `teacher-models/cifar10_teacher/forward_alignment_report.json`，见 `progress-docs/alignment_report.md` |

讲解重点：

| 讲解点 | 可以怎么说 |
| --- | --- |
| checkpoint 对齐说明什么 | 权重命名、shape 映射和转换逻辑没有明显错误。 |
| forward 对齐说明什么 | 在固定输入下，两套框架的模型计算路径输出接近，说明基础网络、预处理和 EDM wrapper 的数值实现基本正确。 |
| forward 对齐不能说明什么 | 不能单独证明完整训练过程完全一致，因为训练还涉及随机 batch、随机 timestep、随机 noise、优化器状态和 GAN 对抗动态。 |

## 2. CIFAR10 5000 Step 训练 Loss 对齐

说明：generator 相关指标只在 generator update step 上统计。由于 `dfake_gen_update_ratio=5`，5000 step 中 generator 更新约 1000 次；guidance 分支每 step 更新一次，所以是 5000 个点。表中 `last` 是最后一个记录点，仅作参考；GAN 与随机 timestep 会导致单点波动，mean 更适合判断整体尺度。

| 指标 | 分支/含义 | 更新点数 | PyTorch mean | Jittor mean | mean 相对差 | PyTorch last | Jittor last | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `generator/loss_dm` | Generator 的 DMD 分布匹配主损失 | 1000 | `0.0399357` | `0.0391488` | `1.97%` | `0.0340053` | `0.0360871` | 核心 DMD loss 尺度高度接近 |
| `generator/gen_cls_loss` | Generator 侧 GAN fooling loss | 1000 | `2.7068744` | `2.8846106` | `6.57%` | `3.0910420` | `2.7874856` | 同一量级，GAN 分支允许更大波动 |
| `loss_generator` | Generator 总 loss，过滤非更新 step | 1000 | `0.0480564` | `0.0478026` | `0.53%` | `0.0432784` | `0.0444496` | 总目标非常接近 |
| `guidance/loss_fake_mean` | Guidance/fake score denoising loss | 5000 | `0.5013468` | `0.4856790` | `3.13%` | `0.6531252` | `0.4605925` | 尺度接近，last 单点波动较大 |
| `guidance/guidance_cls_loss` | Guidance 侧 GAN classifier loss | 5000 | `0.6478751` | `0.6266346` | `3.28%` | `0.6733319` | `0.4930131` | 判别器损失方向和尺度基本一致 |
| `loss_guidance` | Guidance 总 loss | 5000 | `0.5078256` | `0.4919454` | `3.13%` | `0.6598585` | `0.4655226` | 总目标尺度基本一致 |

可以在 PPT 中这样概括：

| 观察 | 解释 |
| --- | --- |
| `loss_generator` mean 相对差约 `0.53%` | Generator 的最终优化目标在两侧几乎同尺度，是最强的训练 loss 对齐证据。 |
| `generator/loss_dm` mean 相对差约 `1.97%` | DMD2 最核心的分布匹配 loss 对齐较好。 |
| Guidance 侧总 loss mean 相对差约 `3.13%` | fake score training 和 classifier 辅助项的整体训练强度接近。 |
| GAN 相关 loss 波动更明显 | GAN 是对抗训练，且未做同 batch、同 noise、同初始化状态的逐步锁定比较，逐点完全重合不现实。 |

## 3. 训练调度与 Loss 公式对齐

| 模块 | PyTorch 侧 | Jittor 侧 | 对齐结论 |
| --- | --- | --- | --- |
| TTUR / generator 更新门控 | `DMD2-pytorch/main/edm/train_edm.py` 中使用 `self.step % self.dfake_gen_update_ratio == 0` | `DMD2-jittor/code/trainer/engine.py` 中使用 `self.global_step % self.dfake_gen_update_ratio == 0` | 语义一致。当前 CIFAR10 正式配置 `dfake_gen_update_ratio=5`，generator 每 5 step 更新一次。 |
| Guidance 更新节奏 | guidance/fake-score/classifier update 每 step 执行 | guidance update 每 step 执行 | 语义一致。 |
| DMD loss | `DMD2-pytorch/main/edm/edm_guidance.py` | `DMD2-jittor/code/models/guidance.py` 与 `DMD2-jittor/code/loss/dmd_loss.py` | 都使用 real/fake denoiser 预测差异构造合成梯度：`grad=(p_real-p_fake)/mean(abs(p_real))`，再用 `0.5*MSE(latents, stop_grad(latents-grad))`。 |
| Fake score loss | `DMD2-pytorch/main/edm/edm_guidance.py` | `DMD2-jittor/code/models/guidance.py` 与 `DMD2-jittor/code/loss/regression_loss.py` | 都对 generator fake image detach 后做 denoising loss，权重形式为 `sigma^-2 + 1/sigma_data^2`。 |
| GAN generator loss | `DMD2-pytorch/main/edm/edm_guidance.py` | `DMD2-jittor/code/models/guidance.py` 与 `DMD2-jittor/code/loss/gan_loss.py` | 都是 `softplus(-D(fake)).mean()`，目标是让 fake 被判为 real。 |
| GAN classifier loss | `DMD2-pytorch/main/edm/edm_guidance.py` | `DMD2-jittor/code/models/guidance.py` 与 `DMD2-jittor/code/loss/gan_loss.py` | 都是 `softplus(D(fake)) + softplus(-D(real))`，目标是真图 logit 变大、假图 logit 变小。 |
| detach/freeze 语义 | generator loss forward 时冻结 guidance 梯度；guidance 训练时 detach fake image | Jittor 当前实现镜像该语义 | 避免 generator step 错误更新 guidance，也避免 guidance step 反向影响 generator。 |

## 4. 目前能证明什么

| 证据 | 支持的结论 | 可信度 |
| --- | --- | --- |
| CIFAR10 teacher forward MAE `2.2411025e-7` | teacher 模型转换和 EDM forward 数值路径基本正确 | 高 |
| ImageNet64 generator forward MAE `5.81759195e-6` | generator checkpoint 转换和 forward 路径基本正确 | 高 |
| `loss_generator` mean 相对差 `0.53%` | Generator 分支整体 loss 组合和更新强度对齐 | 较高 |
| `generator/loss_dm` mean 相对差 `1.97%` | DMD loss 公式、符号和主梯度路径没有明显错位 | 较高 |
| `loss_guidance` mean 相对差 `3.13%` | Guidance 分支 fake score 与 GAN classifier 的整体训练目标接近 | 较高 |
| TTUR 与 loss 公式代码映射 | 控制流和公式实现与 PyTorch 参考设计一致 | 较高 |

## 5. 目前还不能证明什么

| 未完成项 | 为什么重要 | 当前状态 |
| --- | --- | --- |
| 同 seed、同 batch、同 timestep、同 noise 的完整 generator/guidance 单步 loss 对比 | 可以判断单步 forward loss 是否逐数值一致 | 尚未完成 |
| 一次 optimizer step 后的参数 delta 对比 | 可以判断反向传播和优化器更新方向是否一致 | 尚未完成 |
| 同 checkpoint、同 labels、同 noise 的生成样本 pixel 统计对比 | 可以判断推理采样输出是否逐图像接近 | 尚未完成 |
| GAN classifier bottleneck/logit 直接数值对比 | 可以定位 GAN 判别器内部是否完全对齐 | 尚未完成 |
| CIFAR10 正式 FID/IS 对齐 | 可以证明最终生成质量指标是否对齐 | 当前记录中未找到正式结果 |

## 6. 汇报建议

| PPT 页面 | 建议标题 | 展示内容 | 讲解重点 |
| --- | --- | --- | --- |
| 1 | 对齐证据总览 | 放“Forward / Checkpoint 数值对齐”表 | 先证明基础模型和权重转换没有明显问题。 |
| 2 | CIFAR10 Loss 对齐 | 放“5000 Step 训练 Loss 对齐”表 | 强调 `loss_generator`、`generator/loss_dm`、`loss_guidance` 的 mean 相对差较小。 |
| 3 | 为什么 loss 能对齐 | 放“训练调度与 Loss 公式对齐”表 | 解释 TTUR、DMD loss、fake score loss、GAN loss 的代码语义与 PyTorch 对应关系。 |
| 4 | 结论与边界 | 放“目前能证明什么 / 还不能证明什么” | 主动说明尚未完成 FID 和严格单步数值对齐，避免过度宣称。 |

## 一句话结论

当前记录支持这样的表述：Jittor 侧 DMD2 在 teacher/generator forward、核心 loss 公式、GAN loss、TTUR 更新节奏和 CIFAR10 5000 step 训练 loss 尺度上与 PyTorch 侧基本对齐；但目前还不能声称最终 FID/图像质量已经严格对齐，也还缺少同随机性的单步训练数值对齐实验。
