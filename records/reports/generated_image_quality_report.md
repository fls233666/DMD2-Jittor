# 生成图片质量评估报告

生成时间：2026-07-11 01:06:12

## 结论摘要

- 已补充 PyTorch step 5000 图：`DMD2-records/sample_steps_0_2500_5000/pytorch/pytorch_step_005000.png`，并纳入本报告。
- 使用现有图片完成了本地 FID 评估：从 128x128 sample grid 拆出 32x32 CIFAR10 小图，再用 CIFAR10 train 的 8x8 RGB 像素特征统计计算 `Pixel-FID@8x8`。
- 最新节点的 `Pixel-FID@8x8`：PyTorch step5000=5.7798，Jittor fixed step5000=4.4016，Jittor random step5000=5.4154。该指标越低表示低维像素分布越接近 CIFAR10 train。
- 该 FID 是本地可复现的 lightweight / pixel FID，不是论文中的正式 Inception-FID。正式 Inception-FID 仍需要 50k 张最终生成单图、CIFAR10 FID reference `.npz` 与 Inception 权重。

## 应评估的质量指标

| 指标 | 作用 | 是否适合本项目 | 当前本地状态 |
|---|---|---|---|
| FID / Inception-FID | 衡量生成分布与真实分布的距离，生成模型论文最常用 | 最重要，适合作为正式质量指标 | 正式版未完成；本报告已完成 `Pixel-FID@8x8` 作为本地替代评估 |
| KID | 与 FID 类似，但小样本估计偏差更可控 | 可作为 FID 补充 | 当前缺少特征提取器和足量最终样本，未做正式 KID |
| Inception Score | 衡量类别可识别性和类别多样性 | CIFAR10 class-conditional 可作为补充 | 当前没有本地分类器/标准评估脚本，未做 |
| Precision / Recall | 区分真实度与覆盖率，检查模式坍塌 | 很有价值 | 需要大量样本和特征空间，当前未做 |
| 类别准确率 / 类别覆盖率 | 检查 class-conditional 生成是否符合标签 | CIFAR10 分支很适合 | 需要 CIFAR10 分类器，当前未做 |
| 亮度、对比度、熵、清晰度、多样性RMSE | 不依赖网络的 sanity check | 适合本地快速检查，不是论文指标 | 本报告已完成 |

## 本次本地评估范围

| 项目 | 数量/状态 |
|---|---:|
| Jittor sample grid | 20 个 grid，每个 grid 为 4x4 CIFAR10 小图 |
| PyTorch 原始 sample grid | 50 个 grid，step 0 到 4900 |
| PyTorch 补充 step5000 grid | 1 个 grid，来自 `generated_step_005000.png` |
| CIFAR10 train 参考图 | 50000 张 |
| CIFAR10 test 参考图 | 10000 张 |
| 本地 Inception 权重 | 0 个 |

## 现有图片 FID 评估

本节的 FID 计算方式为 `Pixel-FID@8x8`：

1. 将每张 128x128 grid 拆成 16 张 32x32 小图。
2. 将每张 32x32 小图平均池化到 8x8，得到 8x8x3=192 维像素特征。
3. 用 CIFAR10 train 的 50000 张图计算参考均值与协方差。
4. 对生成图特征和真实图特征计算 Frechet Distance。

这可以使用当前已有图片完成，但由于样本数很少、特征不是 Inception 特征，因此只能作为本地质量 sanity check。

| 对象 | 小图数 | Pixel-FID@8x8 | 来源图片 | 拆分后小图目录 |
|---|---:|---:|---|---|
| PyTorch supplemented-grid step 5000 | 16 | 5.7798 | `DMD2-records/sample_steps_0_2500_5000/pytorch/pytorch_step_005000.png` | `DMD2-records/generated_image_quality_fid/tiles/pytorch_step5000` |
| Jittor fixed-grid step 5000 | 16 | 4.4016 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/fixed_step_005000.png` | `DMD2-records/generated_image_quality_fid/tiles/jittor_fixed_step5000` |
| Jittor random-grid step 5000 | 16 | 5.4154 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/random_step_005000.png` | `DMD2-records/generated_image_quality_fid/tiles/jittor_random_step5000` |
| PyTorch local-record step 2500 | 16 | 4.5823 | `DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/samples/generated_step_002500.png` | `DMD2-records/generated_image_quality_fid/tiles/pytorch_step2500` |
| Jittor fixed-grid step 2500 | 16 | 4.4840 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/fixed_step_002500.png` | `DMD2-records/generated_image_quality_fid/tiles/jittor_fixed_step2500` |

展示用三节点聚合结果如下。它把 step 0/2500/5000 的展示小图合并计算，样本数略多，但因为混合了不同训练阶段，不能代表最终模型质量。

| 对象 | 小图数 | Pixel-FID@8x8 | 拆分后小图目录 |
|---|---:|---:|---|
| pytorch_display_0_2500_5000 | 48 | 2.1769 | `DMD2-records/generated_image_quality_fid/tiles/pytorch_display_0_2500_5000` |
| jittor_display_0_2500_5000 | 48 | 2.8948 | `DMD2-records/generated_image_quality_fid/tiles/jittor_display_0_2500_5000` |

FID 结果 JSON：`DMD2-records/generated_image_quality_fid/fid_results.json`  
CIFAR10 Pixel-FID@8x8 参考统计：`DMD2-records/generated_image_quality_fid/cifar10_train_pixel8_stats.npz`

## 指标解释

| 指标 | 含义 | 方向 |
|---|---|---|
| Pixel-FID@8x8 | 8x8 RGB 像素特征上的 Frechet Distance | 越低越接近 CIFAR10 train 的低维像素分布；不是正式 Inception-FID |
| 亮度均值 / 亮度std | 灰度亮度的均值和标准差 | 应接近真实 CIFAR10；过低偏暗，过高偏亮 |
| 像素std | RGB 像素整体标准差 | 太低通常偏灰/模糊，太高可能噪声重 |
| 熵(bits) | 灰度直方图信息量 | 太低说明颜色/纹理单一；过高也可能是噪声 |
| 清晰度LapVar | 灰度 Laplacian 方差 | 太低偏糊，过高可能有噪声或棋盘纹 |
| 多样性RMSE | grid 内小图两两像素 RMSE 均值 | 越高通常多样性越强，但不是语义多样性 |
| 最近真实RMSE | 8x8 下采样特征到 CIFAR10 train 最近邻 RMSE | 过高说明离真实分布远；过低需要警惕记忆，但本指标很粗糙 |

## CIFAR10 真实参考统计

| 参考集 | 图片数 | 亮度均值 | 亮度std | 像素std | 熵(bits) | 清晰度LapVar | 多样性RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| CIFAR10 train | 50000 | 0.4809 | 0.2392 | 0.2516 | 7.8734 | 0.034072 | 0.3416 |
| CIFAR10 test | 10000 | 0.4839 | 0.2387 | 0.2512 | 7.8722 | 0.034014 | 0.3428 |

测试集中 1000 张图到 train set 的最近真实 RMSE-8x8 均值为 `0.1218`，可作为“真实 CIFAR 图像到训练分布”的粗略参照。

## 最新可用生成图评估

| 对象 | step | 小图数 | 亮度均值 | 亮度std | 像素std | 熵(bits) | 清晰度LapVar | 多样性RMSE | 最近真实RMSE | Pixel-FID@8x8 | FD-color | 文件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch supplemented-grid | 5000 | 16 | 0.4056 | 0.2472 | 0.2637 | 7.7748 | 0.038767 | 0.3649 | 0.1295 | 5.7798 | 0.021812 | `DMD2-records/sample_steps_0_2500_5000/pytorch/pytorch_step_005000.png` |
| Jittor fixed-grid | 5000 | 16 | 0.4778 | 0.2443 | 0.2568 | 7.8079 | 0.032755 | 0.3554 | 0.1272 | 4.4016 | 0.005791 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/fixed_step_005000.png` |
| Jittor random-grid | 5000 | 16 | 0.4416 | 0.2699 | 0.2738 | 7.8975 | 0.037012 | 0.3753 | 0.1390 | 5.4154 | 0.013329 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/random_step_005000.png` |

## 关键训练节点评估

| 对象 | step | 小图数 | 亮度均值 | 亮度std | 像素std | 熵(bits) | 清晰度LapVar | 多样性RMSE | 最近真实RMSE | Pixel-FID@8x8 | FD-color | 文件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch local-record | 0 | 16 | 0.4906 | 0.0566 | 0.0697 | 5.8032 | 0.000314 | 0.0831 | 0.0460 | 6.3844 | 0.115385 | `DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/samples/generated_step_000000.png` |
| PyTorch local-record | 2500 | 16 | 0.5099 | 0.2284 | 0.2409 | 7.7306 | 0.021647 | 0.3302 | 0.1090 | 4.5823 | 0.009580 | `DMD2-pytorch/records/cifar10_dmd2_gan_5000_ckpt/samples/generated_step_002500.png` |
| PyTorch supplemented-grid | 5000 | 16 | 0.4056 | 0.2472 | 0.2637 | 7.7748 | 0.038767 | 0.3649 | 0.1295 | 5.7798 | 0.021812 | `DMD2-records/sample_steps_0_2500_5000/pytorch/pytorch_step_005000.png` |
| Jittor fixed-grid | 500 | 16 | 0.5071 | 0.2415 | 0.2467 | 7.8409 | 0.012826 | 0.3364 | 0.1212 | 4.1241 | 0.006984 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/fixed_step_000500.png` |
| Jittor fixed-grid | 2500 | 16 | 0.5138 | 0.2387 | 0.2520 | 7.7497 | 0.031368 | 0.3466 | 0.1289 | 4.4840 | 0.006449 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/fixed_step_002500.png` |
| Jittor fixed-grid | 5000 | 16 | 0.4778 | 0.2443 | 0.2568 | 7.8079 | 0.032755 | 0.3554 | 0.1272 | 4.4016 | 0.005791 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/fixed_step_005000.png` |
| Jittor random-grid | 500 | 16 | 0.4750 | 0.2351 | 0.2447 | 7.7721 | 0.012522 | 0.3396 | 0.1268 | 4.3241 | 0.007543 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/random_step_000500.png` |
| Jittor random-grid | 2500 | 16 | 0.4698 | 0.2630 | 0.2665 | 7.9238 | 0.036609 | 0.3683 | 0.1360 | 4.8787 | 0.007067 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/random_step_002500.png` |
| Jittor random-grid | 5000 | 16 | 0.4416 | 0.2699 | 0.2738 | 7.8975 | 0.037012 | 0.3753 | 0.1390 | 5.4154 | 0.013329 | `DMD2-jittor/outputs/samples/cifar10_dmd2_5000/random_step_005000.png` |

## 对当前结果的解读

1. 补充 PyTorch step5000 后，推荐展示节点可以统一为 PyTorch `0/2500/5000` 对比 Jittor `0/2500/5000`。
2. 从本地 `Pixel-FID@8x8` 看，Jittor fixed step5000 的低维像素分布更接近 CIFAR10 train；PyTorch step5000 的数值高于 PyTorch step2500，说明这批 16 张图的低维分布距离更大。但每个 grid 只有 16 张小图，结论只适合做展示级 sanity check。
3. Jittor fixed-grid 和 random-grid 的含义不同：fixed-grid 用固定输入观察训练变化；random-grid 更接近随机采样展示。二者不应直接理解为同一评估协议下的严格优劣。
4. 若要得到论文级结论，需要生成 Jittor/PyTorch 各 50k 张最终单图，并使用 Inception-FID 或 clean-fid 计算正式指标。

## 可复现的正式评估建议

正式 CIFAR10 质量评估建议按以下优先级补齐：

1. 生成 Jittor 与 PyTorch 各 50k 张最终单图，而不是只保存 4x4 grid。
2. 准备 CIFAR10 FID reference `.npz` 和 Inception detector 权重。
3. 使用 `DMD2-pytorch/third_party/edm/fid.py` 或等价脚本计算 Inception-FID。
4. 若要展示 class-conditional 效果，再补一个 CIFAR10 分类器，统计生成图的类别准确率和类别覆盖率。

本次完整机器可读结果已保存到：`DMD2-records/generated_image_quality_metrics.json`。
