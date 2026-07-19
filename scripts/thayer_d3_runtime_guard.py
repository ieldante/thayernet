"""Two-phase exact-path runtime guard for Thayer-D3B.

The module is standard-library-only.  A bootstrap phase permits package
lifecycle operations only inside a fresh disposable runtime tree.  A strict
phase forbids deletion and cache writes while allowing exact scientific reads
and preregistered fresh-run outputs.  A shutdown phase re-enables cleanup only
inside the disposable runtime tree after readiness evidence is frozen.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import sysconfig
import threading
import traceback
from typing import Any, Iterable, Sequence


class GuardViolation(PermissionError):
    """Raised synchronously when an operation violates the active phase."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(value: os.PathLike[str] | str) -> str:
    return os.path.abspath(os.path.expanduser(os.fspath(value)))


def _under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _fd_path(fd: int) -> str | None:
    """Resolve a directory descriptor used by stdlib scratch cleanup."""

    try:
        return _absolute(os.readlink(f"/dev/fd/{fd}"))
    except (OSError, TypeError, ValueError):
        try:
            value = fcntl.fcntl(fd, 50, b"\0" * 1024)
            resolved = value.split(b"\0", 1)[0].decode("utf-8")
            return _absolute(resolved) if resolved else None
        except (OSError, TypeError, ValueError, UnicodeDecodeError):
            return None


def _path_at(value: os.PathLike[str] | str, dir_fd: Any = None) -> str | None:
    path = os.fspath(value)
    if os.path.isabs(path) or not isinstance(dir_fd, int):
        return _absolute(path)
    parent = _fd_path(dir_fd)
    return _absolute(os.path.join(parent, path)) if parent is not None else None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _stack() -> list[dict[str, Any]]:
    frames = traceback.extract_stack(limit=24)[:-2]
    return [
        {"file": frame.filename, "line": frame.lineno, "function": frame.name}
        for frame in frames
        if not frame.filename.endswith("thayer_d3_runtime_guard.py")
    ][-16:]


def _mode_writes(mode: Any) -> bool:
    if isinstance(mode, str):
        return any(character in mode for character in "wax+")
    if isinstance(mode, int):
        flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(mode & flags)
    return False


@dataclass(frozen=True)
class GuardPolicy:
    repository_root: Path
    fresh_run_root: Path
    runtime_root: Path
    access_log: Path
    blocked_log: Path
    exact_read_files: tuple[Path, ...]
    strict_write_roots: tuple[Path, ...]
    strict_atomic_roots: tuple[Path, ...]
    bootstrap_write_roots: tuple[Path, ...] = ()
    bootstrap_read_roots: tuple[Path, ...] = ()
    protected_markers: tuple[str, ...] = ("atlas", "development", "lockbox")


class TwoPhaseGuard:
    """Process-local audit-hook guard with explicit lifecycle phases."""

    def __init__(self, policy: GuardPolicy) -> None:
        self.policy = policy
        self.phase = "bootstrap"
        self._installed = False
        self._local = threading.local()
        self.event_count = 0
        self.phase_counts: dict[str, int] = {"bootstrap": 0, "strict": 0, "shutdown": 0}
        self._exact_reads = {_absolute(path) for path in policy.exact_read_files}
        self._strict_writes = [_absolute(path) for path in policy.strict_write_roots]
        self._strict_atomic = [_absolute(path) for path in policy.strict_atomic_roots]
        self._repo = _absolute(policy.repository_root)
        self._run = _absolute(policy.fresh_run_root)
        self._runtime = _absolute(policy.runtime_root)
        configured_bootstrap_roots = policy.bootstrap_write_roots or (policy.runtime_root,)
        self._bootstrap_roots = [_absolute(path) for path in configured_bootstrap_roots]
        self._bootstrap_read_roots = [_absolute(path) for path in policy.bootstrap_read_roots]
        self._access_log = _absolute(policy.access_log)
        self._blocked_log = _absolute(policy.blocked_log)
        roots = set(value for value in sysconfig.get_paths().values() if value)
        roots.update((sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)))
        self._environment_roots = sorted({_absolute(root) for root in roots})

    def _is_internal_log(self, path: str) -> bool:
        return path in {self._access_log, self._blocked_log}

    def _protected(self, path: str) -> bool:
        components = {component.casefold() for component in Path(path).parts}
        return any(marker.casefold() in components for marker in self.policy.protected_markers)

    def _system_read(self, path: str) -> bool:
        if path in {"/dev/null", "/dev/urandom", "/dev/random"}:
            return True
        return any(_under(path, root) for root in self._environment_roots)

    def _bootstrap_scratch(self, path: str) -> bool:
        return any(_under(path, root) for root in self._bootstrap_roots)

    def decide(
        self,
        path_value: os.PathLike[str] | str,
        *,
        write: bool = False,
        directory_iteration: bool = False,
    ) -> tuple[bool, str, str]:
        path = _absolute(path_value)
        if self._is_internal_log(path):
            return True, "guard_internal_log", path
        if self._protected(path):
            return False, "protected_partition_marker", path
        if directory_iteration:
            if self.phase in {"bootstrap", "shutdown"} and self._bootstrap_scratch(path):
                return True, "runtime_scratch_iteration", path
            if self.phase in {"bootstrap", "shutdown"} and any(
                _under(path, root) for root in self._bootstrap_read_roots
            ):
                return True, "preregistered_bootstrap_read_iteration", path
            if self._system_read(path):
                return True, "python_environment_iteration", path
            return False, "directory_iteration_denied", path
        if write:
            if self.phase in {"bootstrap", "shutdown"}:
                if self._bootstrap_scratch(path):
                    return True, f"{self.phase}_runtime_write", path
                return False, f"{self.phase}_write_outside_runtime", path
            if any(_under(path, root) for root in self._strict_writes):
                return True, "strict_preregistered_write", path
            return False, "strict_write_not_preregistered", path
        if _under(path, self._run):
            return True, "fresh_run_read", path
        if path in self._exact_reads:
            return True, "exact_allowlisted_read", path
        if self.phase in {"bootstrap", "shutdown"} and any(
            _under(path, root) for root in self._bootstrap_read_roots
        ):
            return True, "preregistered_bootstrap_read", path
        if self._system_read(path):
            return True, "python_environment_read", path
        if _under(path, self._repo):
            return False, "repository_read_not_allowlisted", path
        return False, "external_read_not_allowlisted", path

    def _append(self, target: str, record: dict[str, Any]) -> None:
        if getattr(self._local, "logging", False):
            return
        self._local.logging = True
        try:
            with builtins.open(target, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
        finally:
            self._local.logging = False

    def log(
        self,
        event: str,
        args: Iterable[Any],
        allowed: bool,
        reason: str,
        *,
        path: str | None = None,
        paths: Sequence[str] | None = None,
        include_stack: bool = False,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp_utc": _utcnow(),
            "pid": os.getpid(),
            "phase": self.phase,
            "event": event,
            "args": _jsonable(tuple(args)),
            "allowed": allowed,
            "reason": reason,
            "path": path,
        }
        if paths is not None:
            record["paths"] = list(paths)
        if include_stack:
            record["call_stack"] = _stack()
        self.event_count += 1
        self.phase_counts[self.phase] = self.phase_counts.get(self.phase, 0) + 1
        self._append(self._access_log, record)
        if not allowed:
            self._append(self._blocked_log, record)

    def check(
        self,
        path_value: os.PathLike[str] | str,
        *,
        write: bool = False,
        directory_iteration: bool = False,
        operation: str = "path",
        include_stack: bool = False,
    ) -> str:
        allowed, reason, path = self.decide(
            path_value,
            write=write,
            directory_iteration=directory_iteration,
        )
        self.log(
            operation,
            (os.fspath(path_value),),
            allowed,
            reason,
            path=path,
            include_stack=include_stack,
        )
        if not allowed:
            raise GuardViolation(f"{operation} blocked: {reason}: {path}")
        return path

    def _rename_decision(self, event: str, args: tuple[Any, ...]) -> None:
        values = [value for value in args[:2] if not isinstance(value, int)]
        if len(values) != 2:
            raise GuardViolation(f"{event} requires two pathname arguments")
        source = _path_at(values[0], args[2] if len(args) > 2 else None)
        destination = _path_at(values[1], args[3] if len(args) > 3 else None)
        if source is None or destination is None:
            self.log(event, values, False, "rename_dir_fd_unresolved", include_stack=True)
            raise GuardViolation(f"{event} blocked: unresolved directory descriptor")
        allowed = False
        reason = "rename_denied"
        if self.phase in {"bootstrap", "shutdown"}:
            allowed = self._bootstrap_scratch(source) and self._bootstrap_scratch(destination)
            reason = "runtime_internal_rename" if allowed else "rename_outside_runtime"
        elif self.phase == "strict":
            allowed = any(
                _under(source, root) and _under(destination, root)
                for root in self._strict_atomic
            )
            reason = "strict_preregistered_atomic_rename" if allowed else "strict_rename_not_preregistered"
        self.log(
            event,
            values,
            allowed,
            reason,
            paths=(source, destination),
            include_stack=True,
        )
        if not allowed:
            raise GuardViolation(f"{event} blocked: {reason}: {source} -> {destination}")

    def _delete_decision(self, event: str, args: tuple[Any, ...]) -> None:
        if not args or isinstance(args[0], int):
            raise GuardViolation(f"{event} missing pathname")
        path = _path_at(args[0], args[1] if len(args) > 1 else None)
        if path is None:
            self.log(event, args[:1], False, "delete_dir_fd_unresolved", include_stack=True)
            raise GuardViolation(f"{event} blocked: unresolved directory descriptor")
        allowed = self.phase in {"bootstrap", "shutdown"} and self._bootstrap_scratch(path)
        reason = f"{self.phase}_runtime_delete" if allowed else f"{self.phase}_delete_denied"
        self.log(event, args[:1], allowed, reason, path=path, include_stack=True)
        if not allowed:
            raise GuardViolation(f"{event} blocked: {reason}: {path}")

    def _audit(self, event: str, args: tuple[Any, ...]) -> None:
        if getattr(self._local, "logging", False):
            return
        if event == "open" and args and not isinstance(args[0], int):
            write = _mode_writes(args[1] if len(args) > 1 else "r") or _mode_writes(
                args[2] if len(args) > 2 else 0
            )
            self.check(args[0], write=write, operation="open")
            return
        if event in {"os.listdir", "os.scandir"} and args and args[0] is not None:
            value = _fd_path(args[0]) if isinstance(args[0], int) else _absolute(args[0])
            if value is None:
                self.log(event, args[:1], False, "directory_fd_unresolved", include_stack=True)
                raise GuardViolation(f"{event} blocked: unresolved directory descriptor")
            self.check(value, directory_iteration=True, operation=event)
            return
        if event in {"glob.glob", "glob.glob/2"} and args:
            self.check(args[0], directory_iteration=True, operation=event)
            return
        if event in {"os.rename", "os.replace"}:
            self._rename_decision(event, args)
            return
        if event in {"os.remove", "os.rmdir", "os.unlink"}:
            self._delete_decision(event, args)
            return
        if event == "import" and args:
            module_name = str(args[0])
            allowed = not (self.phase == "strict" and module_name.split(".", 1)[0] == "matplotlib")
            reason = "import_recorded" if allowed else "strict_matplotlib_import_denied"
            self.log(event, (module_name,), allowed, reason)
            if not allowed:
                raise GuardViolation(f"strict Matplotlib import blocked: {module_name}")
            return
        if event == "subprocess.Popen":
            executable = args[0] if args else None
            argv = args[1] if len(args) > 1 else None
            rendered = argv if isinstance(argv, str) else " ".join(
                shlex.quote(str(item)) for item in (argv or ())
            )
            forbidden = ("find ", "rg -R", "grep -R", "ls -R", "du -a", "tree", "**")
            allowed = not any(pattern in rendered for pattern in forbidden)
            reason = "subprocess_pattern_check_pass" if allowed else "recursive_subprocess_denied"
            self.log(event, (executable, argv), allowed, reason)
            if not allowed:
                raise GuardViolation(f"subprocess blocked: {reason}: {rendered}")

    def install(self) -> None:
        if self._installed:
            return
        sys.addaudithook(self._audit)
        self._installed = True
        self.log("guard.install", (), True, "bootstrap_guard_installed")

    def transition(self, phase: str) -> None:
        if phase not in {"strict", "shutdown"}:
            raise ValueError(f"invalid guard transition: {phase}")
        if phase == "strict" and self.phase != "bootstrap":
            raise GuardViolation(f"invalid transition {self.phase} -> strict")
        if phase == "shutdown" and self.phase not in {"bootstrap", "strict"}:
            raise GuardViolation(f"invalid transition {self.phase} -> shutdown")
        prior = self.phase
        self.phase = phase
        self.log(
            "guard.phase_transition",
            (prior, phase),
            True,
            f"phase_transition_{prior}_to_{phase}",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "event_count": self.event_count,
            "phase_counts": dict(self.phase_counts),
        }
