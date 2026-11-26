"""
Results layout. A results folder is split so the part analysis reads can be copied on its own:

    <results>/runs/<run>/        args, returns, metrics.csv.gz, dense.csv.gz, run.log, checkpoint.json
    <results>/artifacts/<run>/   checkpoint.pt, model snapshots, videos, learning-curve PNG

A run folder that is not inside a `runs` folder keeps everything in one place (the old layout).
CSVs are written plain while a run is going and gzipped when it stops; a resumed run unpacks them first.
"""
import glob, gzip, os, shutil

RUNS, ARTIFACTS = "runs", "artifacts"


def artifacts_dir(run_dir):
    """<results>/runs/<run> -> <results>/artifacts/<run>; any other folder is its own artifacts folder"""
    parent, name = os.path.split(os.path.normpath(run_dir))
    if os.path.basename(parent) == RUNS:
        return os.path.join(os.path.dirname(parent), ARTIFACTS, name)
    return run_dir


def find(run_dir, suffix):
    """The run's *<suffix> file, plain or gzipped (plain wins: it is the newer one), or None"""
    for pattern in (f"*{suffix}", f"*{suffix}.gz"):
        files = sorted(glob.glob(os.path.join(run_dir, pattern)))
        if files:
            return files[0]
    return None


def open_text(path):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, newline="")


def gzip_file(path):
    """path -> path.gz, atomically; no-op if path is missing"""
    if not os.path.exists(path):
        return
    with open(path, "rb") as src, gzip.open(path + ".gz.tmp", "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    os.replace(path + ".gz.tmp", path + ".gz")
    os.remove(path)


def gunzip_file(path):
    """path.gz -> path, so it can be appended to; an existing plain file is kept as it is"""
    if os.path.exists(path + ".gz") and not os.path.exists(path):
        with gzip.open(path + ".gz", "rb") as src, open(path + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.replace(path + ".tmp", path)
    if os.path.exists(path + ".gz") and os.path.exists(path):
        os.remove(path + ".gz")
