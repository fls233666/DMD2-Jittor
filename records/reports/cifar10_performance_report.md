# CIFAR10 PyTorch-Jittor 性能测试报告

生成日期：2026-07-11  
图片与曲线来源：`/home/koishi/DMD2/DMD2-jittor/records`

## 结论摘要

| 结论项 | 当前结果 | 说明 |
| --- | --- | --- |
| 核心吞吐量 | PyTorch post-warmup median `37.3874` samples/s；Jittor post-warmup median `50.6978` samples/s | 当前记录中 Jittor 吞吐约为 PyTorch 的 `1.36x`。 |
| 单步总耗时 | PyTorch post-warmup median `0.8559`s；Jittor post-warmup median `0.6312`s | 当前 Jittor 单步总耗时更低，PyTorch/Jittor median total_time 比约 `1.36x`。 |
| 数据加载开销 | PyTorch data_time mean `0.0070`s；Jittor data_time mean `0.0041`s | 两侧 post-warmup data_time 都很小，主要差异来自 step 计算。 |
| TTUR 对耗时影响 | generator update step 慢于 guidance-only step | 两侧都符合 DMD2 每 5 step 更新 generator 的训练结构。 |
| 显存/GPU 指标 | PyTorch 有 torch memory；Jittor 有 nvidia-smi GPU 利用率/显存/功耗 | 两侧来源不同，本报告按各自采样口径分别记录。 |

## 数据来源

| 类型 | 文件 |
| --- | --- |
| PyTorch performance | `../logs/pytorch_cifar10_5000_performance.jsonl` |
| Jittor performance | `../logs/jittor_cifar10_5000_performance.jsonl` |
| PyTorch train metrics | `../logs/pytorch_cifar10_5000_train_metrics.jsonl` |
| Jittor train metrics | `../logs/jittor_cifar10_5000_train_metrics.jsonl` |
| 统计 JSON | `../performance/cifar10_performance_summary.json` |

## 性能曲线

| 图 | 文件 | 说明 |
| --- | --- | --- |
| 吞吐量对比 | `../performance/samples_per_second_compare_ma50.svg` | `samples_per_second`，越高越快，MA50 平滑。 |
| step_time 对比 | `../performance/step_time_compare_ma50.svg` | 纯训练 step 耗时，越低越快，MA50 平滑。 |
| total_time 对比 | `../performance/total_time_compare_ma50.svg` | data + step 的端到端单步耗时，越低越快。 |
| Generator vs Guidance step | `../performance/generator_vs_guidance_step_bars.svg` | 按 `compute_generator_gradient` 拆分典型耗时/吞吐。 |
| PyTorch 显存曲线 | `../performance/pytorch_memory_ma50.svg` | PyTorch CUDA allocated/reserved/step peak memory。 |
| Jittor GPU 利用率 | `../performance/jittor_gpu_utilization_ma50.svg` | 从 nvidia-smi 采样对齐到 Jittor step。 |
| Jittor 显存/功耗 | `../performance/jittor_gpu_memory_power_ma50.svg` | 从 nvidia-smi 采样对齐到 Jittor step。 |

![吞吐量对比](../performance/samples_per_second_compare_ma50.svg)

![step_time 对比](../performance/step_time_compare_ma50.svg)

![total_time 对比](../performance/total_time_compare_ma50.svg)

![Generator vs Guidance step](../performance/generator_vs_guidance_step_bars.svg)

![PyTorch 显存曲线](../performance/pytorch_memory_ma50.svg)

![Jittor GPU 利用率](../performance/jittor_gpu_utilization_ma50.svg)

![Jittor 显存与功耗](../performance/jittor_gpu_memory_power_ma50.svg)

## 核心性能统计

### samples_per_second

| 框架 | count | mean | median | P90 | P95 | min | max | post-warmup mean | post-warmup median | 单位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch | 5000 | 40.0882 | 37.2244 | 60.4092 | 66.3000 | 6.4299 | 104.8192 | 40.1631 | 37.3874 | samples/s |
| Jittor | 5000 | 46.5422 | 50.7013 | 51.5481 | 51.7047 | 8.3582 | 52.1377 | 46.5524 | 50.6978 | samples/s |

### total_time

| 框架 | count | mean | median | P90 | P95 | min | max | post-warmup mean | post-warmup median | 单位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch | 5000 | 0.9187 | 0.8597 | 1.4266 | 1.6498 | 0.3053 | 4.9768 | 0.9162 | 0.8559 | s/step |
| Jittor | 5000 | 0.7220 | 0.6311 | 1.0865 | 1.0930 | 0.6138 | 3.8286 | 0.7214 | 0.6312 | s/step |

### step_time

| 框架 | count | mean | median | P90 | P95 | min | max | post-warmup mean | post-warmup median | 单位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch | 5000 | 0.9116 | 0.8526 | 1.4183 | 1.6432 | 0.2963 | 4.7421 | 0.9092 | 0.8459 | s/step |
| Jittor | 5000 | 0.7180 | 0.6271 | 1.0824 | 1.0889 | 0.6098 | 3.8065 | 0.7173 | 0.6271 | s/step |

### data_time

| 框架 | count | mean | median | P90 | P95 | min | max | post-warmup mean | post-warmup median | 单位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch | 5000 | 0.0070 | 0.0058 | 0.0109 | 0.0133 | 0.0006 | 0.6835 | 0.0070 | 0.0058 | s/step |
| Jittor | 5000 | 0.0041 | 0.0041 | 0.0042 | 0.0042 | 0.0037 | 0.0220 | 0.0041 | 0.0041 | s/step |

## Generator Update vs Guidance-only

DMD2 CIFAR10 训练使用 `dfake_gen_update_ratio=5`，所以每 5 step 更新一次 generator，其余 step 只更新 guidance/fake-score/GAN classifier。下表使用 median 表示典型耗时。

| 框架 | step 类型 | count | median step_time | median total_time | median samples/s |
| --- | --- | ---: | ---: | ---: | ---: |
| PyTorch | generator update | 1000 | 1.4188 | 1.4241 | 22.4708 |
| PyTorch | guidance-only | 4000 | 0.7801 | 0.7860 | 40.7107 |
| Jittor | generator update | 1000 | 1.0824 | 1.0865 | 29.4524 |
| Jittor | guidance-only | 4000 | 0.6243 | 0.6283 | 50.9271 |

解读：generator update step 需要计算 DMD loss 和更新 generator，耗时高于 guidance-only step。曲线中的周期性慢 step 来自该训练结构。

## 总耗时估算

| 框架 | performance 记录数 | step 范围 | step_time 总和 | total_time 总和 | 说明 |
| --- | ---: | --- | ---: | ---: | --- |
| PyTorch | 5000 | 0..4999 | 4558.12s | 4593.26s | 单进程，batch_size_global=32。 |
| Jittor | 5000 | 1..5000 | 3589.85s | 3610.22s | 单进程，batch_size=32。 |

## 显存与 GPU 占用

### PyTorch CUDA memory

| 指标 | mean | median | P95 | max | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `torch_memory_allocated_mb` | 2412.6785 | 2412.6787 | 2412.6787 | 2412.6787 | 当前 step 结束后仍被 tensor 占用的显存。 |
| `torch_step_peak_memory_allocated_mb` | 8910.7809 | 8910.6157 | 8912.1177 | 8912.1177 | 单 step 内 PyTorch allocated 峰值。 |
| `torch_memory_reserved_mb` | 9287.8668 | 9288.0000 | 9288.0000 | 9288.0000 | PyTorch caching allocator 预留显存。 |

### Jittor nvidia-smi GPU monitor

| 指标 | mean | median | P95 | max | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| `gpu_utilization_percent` | 80.1720 | 100.0000 | 100.0000 | 100.0000 | nvidia-smi 对齐到 Jittor step 的 GPU 利用率。 |
| `gpu_memory_used_mib` | 16400.4618 | 16401.0000 | 16401.0000 | 16401.0000 | nvidia-smi 进程/设备级显存占用。 |
| `gpu_power_draw_w` | 171.5378 | 171.4133 | 179.3137 | 183.8800 | nvidia-smi 功耗采样。 |

## 口径说明

- 当前报告覆盖 CIFAR10 DMD2 5000 step 性能日志。
- 两侧都是 batch size 32、单进程记录，具备基础可比性。
- PyTorch 的显存来自 `torch.cuda` allocator 字段；Jittor 的显存来自 nvidia-smi 设备级采样。两者采样来源不同，因此本报告按来源分别记录。
- Jittor GPU 利用率、显存与功耗曲线来自外部 nvidia-smi 采样后按时间戳对齐，和框架内部计时不是同一来源。
