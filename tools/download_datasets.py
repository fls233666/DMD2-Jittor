"""Download datasets used by the DMD2 Jittor debug scripts."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request


def setup_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(project_root, "code")
    for path in (project_root, code_dir, os.path.join(code_dir, "datasets")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return project_root


PROJECT_ROOT = setup_paths()


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILENAME = "cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR10_EXTRACTED_FILES = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}


def file_md5(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def has_expected_md5(path, expected_md5):
    return os.path.exists(path) and file_md5(path) == expected_md5


def run_command(command):
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def format_bytes(num_bytes):
    num_bytes = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(num_bytes) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{num_bytes:.0f}{unit}"
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"


def format_duration(seconds):
    if seconds is None or seconds == float("inf"):
        return "--:--"
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def progress_bar(downloaded, total, start_time, width=42):
    elapsed = max(time.time() - start_time, 1e-8)
    speed = downloaded / elapsed
    if total:
        ratio = min(max(downloaded / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "█" * filled + " " * (width - filled)
        eta = (total - downloaded) / speed if speed > 0 else float("inf")
        return (
            f"{ratio * 100:3.0f}%|{bar}| "
            f"{format_bytes(downloaded)}/{format_bytes(total)} "
            f"[{format_duration(elapsed)}<{format_duration(eta)}, {format_bytes(speed)}/s]"
        )

    return (
        f"{format_bytes(downloaded)} "
        f"[{format_duration(elapsed)}, {format_bytes(speed)}/s]"
    )


def print_progress(downloaded, total, start_time, end=False):
    message = progress_bar(downloaded, total, start_time)
    print(f"\r{message}", end="\n" if end else "", flush=True)


def download_with_python(url, output_path, resume=True, retries=5):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    last_error = None

    for attempt in range(1, int(retries) + 1):
        existing_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        headers = {}
        mode = "wb"
        if resume and existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        request = urllib.request.Request(url, headers=headers)
        start_time = time.time()
        downloaded = existing_size
        total = None

        try:
            print(
                f"Downloading {url} to {output_path} "
                f"(attempt {attempt}/{retries}, resume_from={format_bytes(existing_size)})",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                status = getattr(response, "status", None)
                if mode == "ab" and status == 200:
                    # Server ignored Range; restart from scratch.
                    downloaded = 0
                    mode = "wb"

                content_range = response.headers.get("Content-Range")
                content_length = response.headers.get("Content-Length")
                if content_range and "/" in content_range:
                    total = int(content_range.rsplit("/", 1)[1])
                elif content_length:
                    total = int(content_length) + (downloaded if mode == "ab" else 0)

                with open(output_path, mode) as handle:
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        print_progress(downloaded, total, start_time)

            print_progress(downloaded, total, start_time, end=True)
            return output_path
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"\ndownload attempt {attempt} failed: {exc}", flush=True)
            if attempt < int(retries):
                time.sleep(min(5 * attempt, 30))

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_with_curl(url, output_path, resume=True, retries=5):
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(int(retries)),
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "-o",
        output_path,
    ]
    if resume:
        command[1:1] = ["-C", "-"]
    command.append(url)
    run_command(command)


def download_with_wget(url, output_path, resume=True, retries=5):
    command = [
        "wget",
        "--tries",
        str(int(retries)),
        "--timeout",
        "30",
        "-O",
        output_path,
    ]
    if resume:
        command.insert(1, "-c")
    command.append(url)
    run_command(command)


def check_cifar10_integrity(root, require_archive=False, verbose=True):
    archive_path = os.path.join(root, CIFAR10_FILENAME)
    base_dir = os.path.join(root, "cifar-10-batches-py")
    problems = []

    archive_ok = has_expected_md5(archive_path, CIFAR10_MD5)
    if require_archive and not archive_ok:
        if os.path.exists(archive_path):
            problems.append(
                f"archive md5 mismatch: {archive_path} "
                f"got {file_md5(archive_path)} expected {CIFAR10_MD5}"
            )
        else:
            problems.append(f"archive missing: {archive_path}")

    for filename, expected_md5 in CIFAR10_EXTRACTED_FILES.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            problems.append(f"missing extracted file: {path}")
            continue
        got_md5 = file_md5(path)
        if got_md5 != expected_md5:
            problems.append(
                f"md5 mismatch: {path} got {got_md5} expected {expected_md5}"
            )

    ok = not problems
    if verbose:
        if ok:
            print(f"CIFAR-10 integrity check passed: {root}", flush=True)
        else:
            print(f"CIFAR-10 integrity check failed: {root}", flush=True)
            for problem in problems:
                print(f"  - {problem}", flush=True)
    return ok, problems


def extract_cifar10_archive(root):
    archive_path = os.path.join(root, CIFAR10_FILENAME)
    if not has_expected_md5(archive_path, CIFAR10_MD5):
        raise RuntimeError(f"Cannot extract unverified archive: {archive_path}")

    print(f"Extracting {archive_path} to {root}", flush=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(root)


def prepare_cifar10_archive(root, method="auto", resume=True, retries=5):
    os.makedirs(root, exist_ok=True)
    archive_path = os.path.join(root, CIFAR10_FILENAME)

    if has_expected_md5(archive_path, CIFAR10_MD5):
        print(f"archive already downloaded and verified: {archive_path}", flush=True)
        return archive_path

    if method == "python":
        download_with_python(
            CIFAR10_URL,
            archive_path,
            resume=resume,
            retries=retries,
        )
        if has_expected_md5(archive_path, CIFAR10_MD5):
            print(f"archive verified: {archive_path}", flush=True)
            return archive_path
        got = file_md5(archive_path) if os.path.exists(archive_path) else "missing"
        raise RuntimeError(
            f"Downloaded archive md5 mismatch: got {got}, expected {CIFAR10_MD5}"
        )

    if method == "jittor":
        return archive_path

    methods = []
    if method == "auto":
        methods.append("python")
        if shutil.which("curl"):
            methods.append("curl")
        if shutil.which("wget"):
            methods.append("wget")
    else:
        methods.append(method)

    if not methods:
        raise RuntimeError("Neither curl nor wget is available; use --method jittor.")

    last_error = None
    for current_method in methods:
        try:
            if current_method == "python":
                download_with_python(
                    CIFAR10_URL,
                    archive_path,
                    resume=resume,
                    retries=retries,
                )
            elif current_method == "curl":
                download_with_curl(
                    CIFAR10_URL,
                    archive_path,
                    resume=resume,
                    retries=retries,
                )
            elif current_method == "wget":
                download_with_wget(
                    CIFAR10_URL,
                    archive_path,
                    resume=resume,
                    retries=retries,
                )
            else:
                raise ValueError(f"Unsupported download method: {current_method}")

            if has_expected_md5(archive_path, CIFAR10_MD5):
                print(f"archive verified: {archive_path}", flush=True)
                return archive_path

            got = file_md5(archive_path) if os.path.exists(archive_path) else "missing"
            raise RuntimeError(
                f"Downloaded archive md5 mismatch: got {got}, expected {CIFAR10_MD5}"
            )
        except Exception as exc:
            last_error = exc
            print(f"{current_method} download failed: {exc}", flush=True)

    raise RuntimeError(f"Failed to download CIFAR-10 archive: {last_error}")


def download_cifar10(root, train=True, test=True, method="auto", resume=True, retries=5):
    os.makedirs(root, exist_ok=True)
    prepare_cifar10_archive(
        root=root,
        method=method,
        resume=resume,
        retries=retries,
    )

    extracted_ok, _ = check_cifar10_integrity(root, verbose=False)
    if not extracted_ok and has_expected_md5(os.path.join(root, CIFAR10_FILENAME), CIFAR10_MD5):
        extract_cifar10_archive(root)

    from jittor.dataset.cifar import CIFAR10 as JittorCIFAR10

    if train:
        JittorCIFAR10(root=root, train=True, transform=None, download=True)
    if test:
        JittorCIFAR10(root=root, train=False, transform=None, download=True)
    ok, problems = check_cifar10_integrity(root)
    if not ok:
        raise RuntimeError("CIFAR-10 integrity check failed: " + "; ".join(problems))
    return root


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("cifar10",),
        default="cifar10",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--root",
        default=os.path.join(PROJECT_ROOT, "data", "cifar10"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--method",
        choices=("auto", "python", "curl", "wget", "jittor"),
        default="auto",
        help="Download method. auto uses Python progress download before curl/wget/Jittor.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify downloaded/extracted CIFAR-10 files.",
    )
    parser.add_argument(
        "--require-archive",
        action="store_true",
        help="When checking, require cifar-10-python.tar.gz to be present and md5-valid.",
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    if args.train_only and args.test_only:
        raise ValueError("--train-only and --test-only are mutually exclusive.")

    if args.check_only:
        if args.dataset == "cifar10":
            ok, _ = check_cifar10_integrity(
                args.root,
                require_archive=args.require_archive,
            )
            return 0 if ok else 1
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train = not args.test_only
    test = not args.train_only
    if args.dataset == "cifar10":
        path = download_cifar10(
            args.root,
            train=train,
            test=test,
            method=args.method,
            resume=not args.no_resume,
            retries=args.retries,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"dataset ready: {args.dataset} root={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
