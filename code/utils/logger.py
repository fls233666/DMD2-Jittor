"""Logging helpers for DMD2 debug experiments."""

import csv
import json
import os
import time
from collections.abc import Mapping

import numpy as np


def ensure_dir(path):
    directory = path if os.path.splitext(path)[1] == "" else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def to_python(value):
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            pass

    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {str(key): to_python(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]

    return value


def is_scalar(value):
    value = to_python(value)
    return isinstance(value, (int, float, bool))


def scalar_logs(logs):
    return {
        str(key): float(to_python(value))
        for key, value in logs.items()
        if is_scalar(value)
    }


def write_json(path, data, indent=2):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_python(data), handle, indent=indent, sort_keys=True)
        handle.write("\n")
    return path


def append_jsonl(path, data):
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_python(data), sort_keys=True))
        handle.write("\n")
    return path


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_text(path, text):
    ensure_dir(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(str(text))
        if not str(text).endswith("\n"):
            handle.write("\n")
    return path


class JSONLLogger:
    # Append structured records to a JSONL file.
    def __init__(self, path, include_timestamp=True):
        self.path = path
        self.include_timestamp = include_timestamp
        ensure_dir(path)

    def write(self, record):
        record = dict(record)
        if self.include_timestamp and "time" not in record:
            record["time"] = time.time()
        append_jsonl(self.path, record)

    def close(self):
        pass


class CSVLogger:
    # Append scalar records to a CSV file while keeping a stable header.
    def __init__(self, path):
        self.path = path
        self.fieldnames = None
        ensure_dir(path)

    def write(self, record):
        record = scalar_logs(record)
        if self.fieldnames is None:
            self.fieldnames = list(record.keys())
            file_exists = os.path.exists(self.path) and os.path.getsize(self.path) > 0
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
            return

        row = {name: record.get(name, "") for name in self.fieldnames}
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)

    def close(self):
        pass


def make_logger(path):
    if path is None:
        return None
    if hasattr(path, "write"):
        return path
    if str(path).endswith(".csv"):
        return CSVLogger(path)
    return JSONLLogger(path)


def write_alignment_log(path, report_text, summary=None, append=True):
    record = {
        "time": time.time(),
        "summary": {} if summary is None else to_python(summary),
        "report": str(report_text),
    }

    if str(path).endswith(".jsonl"):
        return append_jsonl(path, record)

    if str(path).endswith(".json"):
        return write_json(path, record)

    mode_text = "a" if append else "w"
    ensure_dir(path)
    with open(path, mode_text, encoding="utf-8") as handle:
        handle.write(f"# alignment report time={record['time']:.3f}\n")
        if summary is not None:
            handle.write(json.dumps(to_python(summary), sort_keys=True))
            handle.write("\n")
        handle.write(str(report_text).rstrip())
        handle.write("\n")
    return path
