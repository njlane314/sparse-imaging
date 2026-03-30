#!/usr/bin/env python3
from __future__ import annotations

import ast
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON_FILES = tuple(sorted(ROOT.glob("*.py")))
SHELL_FILES = (ROOT / ".setup.sh", ROOT / ".workflow.cmds")


class CheckFailure(RuntimeError):
    pass


def _ok(message: str) -> None:
    print(f"[ok] {message}")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def check_python_syntax() -> None:
    for path in PYTHON_FILES:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            raise CheckFailure(f"{path.name}: {exc.msg}") from exc
        _ok(f"py_compile {path.name}")


def check_shell_syntax() -> None:
    for path in SHELL_FILES:
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise CheckFailure(f"{path.name}: {result.stderr.strip() or result.stdout.strip()}")
        _ok(f"bash -n {path.name}")


def _assigned_uppercase_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in _module_ast(path).body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                names.add(target.id)
    return names


def _cfg_names_used(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_module_ast(path)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "cfg":
            names.add(node.attr)
    return names


def check_config_coverage() -> None:
    defined = _assigned_uppercase_names(ROOT / "config.py")
    for path in (ROOT / "process.py", ROOT / "train.py"):
        missing = sorted(_cfg_names_used(path) - defined)
        if missing:
            raise CheckFailure(f"{path.name}: undefined cfg names: {', '.join(missing)}")
    _ok("config.py covers cfg.* references in process.py and train.py")


def check_workflow() -> None:
    workflow = ROOT / ".workflow.cmds"
    lines = [
        line.strip()
        for line in _read(workflow).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    try:
        process_idx = lines.index("python3 -u process.py")
        train_idx = lines.index("python3 -u train.py")
    except ValueError as exc:
        raise CheckFailure(".workflow.cmds must call process.py and train.py with python3") from exc
    if process_idx >= train_idx:
        raise CheckFailure(".workflow.cmds must run process.py before train.py")
    exported = {
        line.split()[1].split("=", 1)[0]
        for line in lines
        if line.startswith("export ") and "=" in line
    }
    required = {"ROOT_FILE", "TREE", "SHARDS_DIR", "CHECKPOINT_PATH", "LOSS_LOG_PATH"}
    missing = sorted(required - exported)
    if missing:
        raise CheckFailure(f".workflow.cmds missing required exports: {', '.join(missing)}")
    for script_name in ("process.py", "train.py"):
        if not (ROOT / script_name).exists():
            raise CheckFailure(f".workflow.cmds references missing file: {script_name}")
    _ok(".workflow.cmds order and required exports")


def _extract_function_source(path: Path, name: str) -> str:
    source = _read(path)
    module = ast.parse(source, filename=str(path))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            snippet = ast.get_source_segment(source, node)
            if snippet is None:
                break
            return snippet
    raise CheckFailure(f"could not locate function {name} in {path.name}")


class _MiniArray:
    def __init__(self, data):
        self._data = list(data)

    @property
    def size(self) -> int:
        return len(self._data)

    def astype(self, _dtype, copy=False):
        return _MiniArray(self._data)

    def tolist(self) -> list[int]:
        return list(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _MiniArray(self._data[item])
        if isinstance(item, _MiniArray):
            item = item._data
        if isinstance(item, list):
            if item and all(isinstance(value, bool) for value in item):
                return _MiniArray(value for value, keep in zip(self._data, item) if keep)
            return _MiniArray(self._data[index] for index in item)
        return self._data[item]

    def __eq__(self, other):
        return [value == other for value in self._data]

    def __ne__(self, other):
        return [value != other for value in self._data]


class _MiniRng:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def permutation(self, n: int) -> list[int]:
        return list(range(int(n) - 1, -1, -1))


class _MiniRandomModule:
    @staticmethod
    def default_rng(seed: int) -> _MiniRng:
        return _MiniRng(seed)


class _MiniNumpy:
    ndarray = _MiniArray
    int64 = int
    random = _MiniRandomModule()

    @staticmethod
    def flatnonzero(mask) -> _MiniArray:
        if isinstance(mask, _MiniArray):
            mask = mask.tolist()
        return _MiniArray(index for index, keep in enumerate(mask) if keep)

    @staticmethod
    def any(values) -> bool:
        if isinstance(values, _MiniArray):
            values = values.tolist()
        return any(values)

    @staticmethod
    def concatenate(parts) -> _MiniArray:
        data = []
        for part in parts:
            if isinstance(part, _MiniArray):
                data.extend(part.tolist())
            else:
                data.extend(list(part))
        return _MiniArray(data)


def check_train_logic() -> None:
    namespace = {"Tuple": tuple, "np": _MiniNumpy}
    exec(_extract_function_source(ROOT / "train.py", "poly_lr"), namespace, namespace)
    exec(_extract_function_source(ROOT / "train.py", "_split_indices"), namespace, namespace)
    poly_lr = namespace["poly_lr"]
    split_indices = namespace["_split_indices"]

    schedule = [poly_lr(step, 10, 0.5, 1.0) for step in range(11)]
    if schedule[0] != 0.5 or schedule[-1] != 0.0:
        raise CheckFailure("train.poly_lr endpoints changed unexpectedly")
    if any(a < b for a, b in zip(schedule, schedule[1:])):
        raise CheckFailure("train.poly_lr should be monotonically non-increasing")
    if poly_lr(3, 0, 0.5, 0.9) != 0.5:
        raise CheckFailure("train.poly_lr should return lr0 when max_steps <= 0")

    train_idx, val_idx = split_indices(_MiniArray([0, 1]), _MiniArray([True, True]), 0.5, 123)
    if train_idx.tolist() != [0, 1] or val_idx.tolist() != []:
        raise CheckFailure("train._split_indices should repair missing training classes")

    labels = _MiniArray([0, 0, 1, 1, 0, 1])
    keep_mask = _MiniArray([True, False, True, True, True, True])
    train_idx, val_idx = split_indices(labels, keep_mask, 0.4, 7)
    used = sorted(train_idx.tolist() + val_idx.tolist())
    if used != [0, 2, 3, 4, 5]:
        raise CheckFailure("train._split_indices should only use kept events")
    train_labels = {labels[index] for index in train_idx.tolist()}
    if train_labels != {0, 1}:
        raise CheckFailure("train._split_indices must leave both classes in training")
    try:
        split_indices(_MiniArray([1]), _MiniArray([True]), 0.5, 0)
    except ValueError:
        pass
    else:
        raise CheckFailure("train._split_indices should reject fewer than two valid events")
    _ok("train.py logic smoke tests")


def main() -> int:
    checks = (
        check_python_syntax,
        check_shell_syntax,
        check_config_coverage,
        check_workflow,
        check_train_logic,
    )
    try:
        for check in checks:
            check()
    except CheckFailure as exc:
        print(f"[fail] {exc}", file=sys.stderr)
        return 1
    print("[ok] local checks completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
