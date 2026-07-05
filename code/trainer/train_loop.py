"""Simple train loop utilities for image DMD2 runs."""

import os

try:
    from .checkpoint import save_checkpoint
except ImportError:
    from checkpoint import save_checkpoint

try:
    from .evaluator import cleanup_jittor_memory
except ImportError:
    try:
        from evaluator import cleanup_jittor_memory
    except ImportError:
        cleanup_jittor_memory = None

try:
    from utils.logger import make_logger, scalar_logs
    from utils.metric import StepTimer, infer_batch_size, performance_record
except ImportError:
    try:
        from ..utils.logger import make_logger, scalar_logs
        from ..utils.metric import StepTimer, infer_batch_size, performance_record
    except ImportError:
        from logger import make_logger, scalar_logs
        from metric import StepTimer, infer_batch_size, performance_record


def format_log(logs):
    parts = []
    for name, value in logs.items():
        if isinstance(value, float):
            parts.append(f"{name}={value:.6g}")
        else:
            parts.append(f"{name}={value}")
    return " ".join(parts)


class MetricAverager:
    # Track running scalar means for lightweight console logging.
    def __init__(self):
        self.totals = {}
        self.counts = {}

    def update(self, logs):
        for name, value in logs.items():
            if not isinstance(value, (int, float)):
                continue
            self.totals[name] = self.totals.get(name, 0.0) + float(value)
            self.counts[name] = self.counts.get(name, 0) + 1

    def mean(self):
        return {
            name: self.totals[name] / max(self.counts[name], 1)
            for name in self.totals
        }


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train_image_dmd2(
    engine,
    train_loader,
    max_steps=1000,
    log_interval=50,
    checkpoint_interval=1000,
    output_dir=None,
    start_step=0,
    evaluator=None,
    eval_interval=None,
    metrics_logger=None,
    performance_logger=None,
    performance_monitor=None,
    print_fn=print,
):
    # Run a compact image DMD2 training loop.
    os.makedirs(output_dir, exist_ok=True) if output_dir else None
    data_iter = cycle(train_loader)
    averager = MetricAverager()
    history = []
    timer = StepTimer()
    metrics_logger = make_logger(metrics_logger)
    performance_logger = make_logger(performance_logger)

    for step in range(int(start_step), int(max_steps)):
        batch = next(data_iter)
        data_time = timer.mark_step_start()
        result = engine.train_step(batch)
        step_time = timer.mark_step_end()

        logs = result["logs"]
        logs["step"] = step + 1
        logs["data_time"] = data_time
        logs["step_time"] = step_time
        logs["samples_per_second"] = performance_record(
            step=step + 1,
            batch_size=infer_batch_size(batch),
            data_time=data_time,
            step_time=step_time,
        )["samples_per_second"]

        perf_extra = None
        if performance_monitor is not None:
            perf_extra = performance_monitor.collect_since_last()

        perf_logs = performance_record(
            step=step + 1,
            batch_size=infer_batch_size(batch),
            data_time=data_time,
            step_time=step_time,
            extra=perf_extra,
        )

        averager.update(logs)
        history.append(dict(logs))
        if metrics_logger is not None:
            metrics_logger.write(scalar_logs(logs))
        if performance_logger is not None:
            performance_logger.write(perf_logs)

        if log_interval and (step + 1) % int(log_interval) == 0:
            print_fn(format_log(averager.mean()))
            averager = MetricAverager()

        if evaluator is not None and eval_interval and (step + 1) % int(eval_interval) == 0:
            if cleanup_jittor_memory is not None:
                cleanup_jittor_memory()
            evaluator.evaluate(engine.model, step=step + 1)
            if cleanup_jittor_memory is not None:
                cleanup_jittor_memory()

        if output_dir and checkpoint_interval and (step + 1) % int(checkpoint_interval) == 0:
            save_checkpoint(
                path=os.path.join(output_dir, f"checkpoint_{step + 1:06d}.pkl"),
                model=engine.model,
                generator_optimizer=engine.generator_optimizer,
                guidance_optimizer=engine.guidance_optimizer,
                generator_scheduler=engine.generator_scheduler,
                guidance_scheduler=engine.guidance_scheduler,
                ema=engine.ema,
                step=step + 1,
            )

        timer.reset_data_start()

    return history


train_debug = train_image_dmd2
