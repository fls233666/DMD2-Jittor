# PyTorch-Jittor 精度对齐情况汇总

## 总体结论

| 结论项 | 当前判断 | 说明 |
| --- | --- | --- |
| forward 数值对齐 | 已有强证据 | CIFAR10 teacher forward 和 ImageNet64 generator forward 都有逐 tensor 误差统计，误差在较小范围内。 |
| CIFAR10 训练 loss 对齐 | 基本对齐 | 5000 step 训练记录中，核心 loss 的均值尺度接近，`summary.json` 中对应曲线状态均为 `ok`。 |
| 训练调度与 loss 公式 | 代码路径对齐 | TTUR、DMD loss、fake score loss、GAN generator loss、GAN classifier loss 的代码语义与 PyTorch 侧一致。 |
| 生成质量检查 | 已有本地代理评估 | 当前 records 中包含 PyTorch/Jittor step 0/2500/5000 采样图与 `Pixel-FID@8x8` 记录。 |

## 数据来源

| 类型 | 文件/路径 | 用途 |
| --- | --- | --- |
| loss 对齐汇总 | `../loss/summary.json` | 汇总 PyTorch 与 Jittor CIFAR10 5000 step 训练 loss 的 count、mean、last、min、max。 |
| Jittor 训练日志 | `../logs/jittor_cifar10_5000_train_metrics.jsonl` | Jittor 侧 CIFAR10 5000 step 原始训练记录。 |
| PyTorch 训练日志 | `../logs/pytorch_cifar10_5000_train_metrics.jsonl` | PyTorch 侧 CIFAR10 5000 step 原始训练记录。 |
| 采样图总览 | `../samples/comparison/pytorch_jittor_training_samples_overview.png` | 展示 PyTorch/Jittor step 0/2500/5000 的生成效果。 |
| 质量评估结果 | `../quality/generated_image_quality_metrics.json` | 本地图像质量代理指标与 `Pixel-FID@8x8`。 |
| FID 结果 | `../quality/fid_results.json` | sample grid 拆分后的 lightweight FID 结果。 |

## 图表索引

![PyTorch-Jittor 训练节点总览](../samples/comparison/pytorch_jittor_training_samples_overview.png)

![Generator loss](../loss/loss_generator_ma20.svg)

![Generator DMD loss](../loss/generator_loss_dm_ma20.svg)

![Guidance loss](../loss/loss_guidance_ma20.svg)

![Fake score loss](../loss/guidance_loss_fake_mean_ma20.svg)

![Generator GAN classifier loss](../loss/generator_gen_cls_loss_ma20.svg)

![Guidance GAN classifier loss](../loss/guidance_guidance_cls_loss_ma20.svg)

## 1. Forward / Checkpoint 数值对齐

| 对齐项 | PyTorch 侧 | Jittor 侧 | 差异/误差 | 判断 | 来源 |
| --- | --- | --- | --- | --- | --- |
| ImageNet64 generator checkpoint 转换 | 输入 key 数 `553` | 转换后 key 数 `541` | 丢弃确定性的 `resample_filter` key `12` 个；unexpected/missing/shape mismatch/duplicate 均为 `0` | 权重映射通过 | `progress-docs/alignment_report.md` |
| ImageNet64 generator forward | output mean/std = `0.15208736 / 0.65460753` | output mean/std = `0.15208854 / 0.65460896` | MAE `5.81759195e-6`；median AE `2.71201134e-6`；max AE `1.75967813e-4`；RMSE `1.26740533e-5` | forward 数值基本对齐 | `progress-docs/alignment_report.md` |
| CIFAR10 teacher EDM forward | output mean/std = `-0.0220896006 / 0.0422849022` | output mean/std = `-0.0220895056 / 0.0422849059` | MAE `2.2411025e-7`；max AE `9.1642141e-7` | teacher forward 高精度对齐 | `teacher-models/cifar10_teacher/forward_alignment_report.json` |

## 2. CIFAR10 5000 Step 训练 Loss 对齐

说明：generator 相关指标只在 generator update step 上统计。由于 `dfake_gen_update_ratio=5`，5000 step 中 generator 更新约 1000 次；guidance 分支每 step 更新一次，所以是 5000 个点。表中 `last` 是最后一个记录点，mean 更适合判断整体尺度。

| 指标 | 分支/含义 | 更新点数 | PyTorch mean | Jittor mean | mean 相对差 | PyTorch last | Jittor last | 判断 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `generator/loss_dm` | Generator 的 DMD 分布匹配主损失 | 1000 | `0.0399357` | `0.0391488` | `1.97%` | `0.0340053` | `0.0360871` | 核心 DMD loss 尺度高度接近 |
| `generator/gen_cls_loss` | Generator 侧 GAN fooling loss | 1000 | `2.7068744` | `2.8846106` | `6.57%` | `3.0910420` | `2.7874856` | 同一量级 |
| `loss_generator` | Generator 总 loss，过滤非更新 step | 1000 | `0.0480564` | `0.0478026` | `0.53%` | `0.0432784` | `0.0444496` | 总目标非常接近 |
| `guidance/loss_fake_mean` | Guidance/fake score denoising loss | 5000 | `0.5013468` | `0.4856790` | `3.13%` | `0.6531252` | `0.4605925` | 尺度接近 |
| `guidance/guidance_cls_loss` | Guidance 侧 GAN classifier loss | 5000 | `0.6478751` | `0.6266346` | `3.28%` | `0.6733319` | `0.4930131` | 判别器损失方向和尺度基本一致 |
| `loss_guidance` | Guidance 总 loss | 5000 | `0.5078256` | `0.4919454` | `3.13%` | `0.6598585` | `0.4655226` | 总目标尺度基本一致 |

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

