"""
Enforce numpy-style argument documentation that ruff cannot check.

Ruff (D417) only validates a ``Parameters`` section that already exists; this
test requires the section itself for every function, method, and class that
takes arguments, in ``src/``, ``serving/src/`` and ``scripts/``.
"""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_ROOTS = ("src", "serving/src", "scripts")
SECTION_NAMES = {
    "Returns",
    "Raises",
    "Yields",
    "Attributes",
    "Notes",
    "Examples",
    "See Also",
    "Warns",
}
PARAM_LINE = re.compile(r"^\*{0,2}[A-Za-z_]\w*(\s*,\s*\*{0,2}[A-Za-z_]\w*)*\s*:")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in CHECKED_ROOTS:
        files.extend(
            path
            for path in (REPO_ROOT / root).rglob("*.py")
            if ".venv" not in path.parts and "proto" not in path.parts
        )
    return sorted(files)


def _documented_params(docstring: str | None) -> set[str] | None:
    """Return the names listed under ``Parameters``, or None without that section."""
    if not docstring:
        return None
    lines = docstring.splitlines()
    for i, line in enumerate(lines[:-1]):
        if line.strip() != "Parameters" or set(lines[i + 1].strip()) != {"-"}:
            continue
        base_indent = len(line) - len(line.lstrip())
        names: set[str] = set()
        for entry in lines[i + 2 :]:
            stripped = entry.strip()
            if not stripped or len(entry) - len(entry.lstrip()) != base_indent:
                continue
            if stripped in SECTION_NAMES:
                break
            if PARAM_LINE.match(stripped):
                names.update(
                    name.strip().lstrip("*") for name in stripped.split(":")[0].split(",")
                )
        return names
    return None


def _arg_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [name for name in names if name not in ("self", "cls")]


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = (
            target.attr
            if isinstance(target, ast.Attribute)
            else getattr(target, "id", "")
        )
        if name == "dataclass":
            return True
    return False


def _dataclass_fields(node: ast.ClassDef) -> list[str]:
    return [
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and "ClassVar" not in ast.unparse(stmt.annotation)
    ]


def _problems(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(REPO_ROOT).as_posix()
    problems: list[str] = []

    def check(node: ast.AST, required: list[str], label: str) -> None:
        if not required:
            return
        documented = _documented_params(ast.get_docstring(node))
        if documented is None:
            problems.append(f"{rel}:{node.lineno} {label}: no Parameters section")
            return
        missing = [name for name in required if name not in documented]
        if missing:
            problems.append(f"{rel}:{node.lineno} {label}: undocumented {missing}")

    def visit(body: list[ast.stmt], owner: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue
                label = f"{owner}.{node.name}" if owner else node.name
                check(node, _arg_names(node.args), label)
            elif isinstance(node, ast.ClassDef):
                init = next(
                    (
                        n
                        for n in node.body
                        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                    ),
                    None,
                )
                if init is not None:
                    check(node, _arg_names(init.args), f"class {node.name}")
                elif _is_dataclass(node):
                    check(node, _dataclass_fields(node), f"dataclass {node.name}")
                visit(node.body, node.name)

    visit(tree.body, None)
    return problems


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_arguments_are_documented(path: Path) -> None:
    """Every argument of every function, method and class is in its Parameters."""
    problems = _problems(path)
    assert not problems, "\n".join(problems)
