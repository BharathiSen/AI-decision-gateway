"""Replay ledger: each authorization's decision_id executes at most once.

MemoryLedger lives as long as its process (Stage 1 in-process boundary).
FileLedger is durable: an append-only JSON-lines file, fsynced before an
execution is allowed, reloaded on start, so a restarted executor still
refuses a decision it already consumed. Entries are kept until well after
they expire; an expired authorization is refused by its validity window
anyway, so dropping it from the file cannot re-enable it.
"""

import contextlib
import json
import os
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: one executor process per ledger is assumed
    fcntl = None

# Keep consumed ids this long after expiry before compaction may drop them.
RETAIN_AFTER_EXPIRY_SECONDS = 3600.0
ALREADY_USED = "decision already used"


def _fsync_dir(path):
    """Make a create/rename durable (POSIX). No-op where directories cannot be opened."""
    if os.name != "posix":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _exclusive(path):
    """Cross-process lock on <ledger>.lock (POSIX flock)."""
    if fcntl is None:
        yield
        return
    with open(str(path) + ".lock", "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LedgerError(RuntimeError):
    """The ledger cannot be read or written. Execution must not proceed."""


class MemoryLedger:
    def __init__(self):
        self._entries = {}
        self._lock = threading.Lock()

    def consume(self, decision_id, expires_at, now=None):
        if not isinstance(decision_id, str) or not decision_id:
            return "authorization has no decision id"
        with self._lock:
            if decision_id in self._entries:
                return ALREADY_USED
            self._entries[decision_id] = float(expires_at or 0.0)
        return None

    def __contains__(self, decision_id):
        return decision_id in self._entries


class FileLedger(MemoryLedger):
    def __init__(self, path, now=None):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path):
            self._load(time.time() if now is None else now)

    def _load(self, now, compact=True):
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LedgerError(f"ledger {self.path}: {exc}") from exc
        kept = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                decision_id = entry["decision_id"]
                expires_at = float(entry["expires_at"])
            except (ValueError, KeyError, TypeError) as exc:
                # A torn last line from a crash is the only expected damage.
                if number == len(lines):
                    continue
                raise LedgerError(f"ledger {self.path}: line {number} unreadable: {exc}") from exc
            self._entries[decision_id] = expires_at
            if expires_at + RETAIN_AFTER_EXPIRY_SECONDS > now:
                kept.append(line)
        if compact and len(kept) != len([line for line in lines if line.strip()]):
            self._rewrite(kept)

    def _rewrite(self, lines):
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write("".join(line + "\n" for line in lines))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)
        _fsync_dir(self.path.parent)

    def consume(self, decision_id, expires_at, now=None):
        if not isinstance(decision_id, str) or not decision_id:
            return "authorization has no decision id"
        with self._lock, _exclusive(self.path):
            # Another executor process may have appended since we loaded.
            self._load(time.time() if now is None else now, compact=False)
            if decision_id in self._entries:
                return ALREADY_USED
            created = not self.path.exists()
            entry = {
                "decision_id": decision_id,
                "expires_at": float(expires_at or 0.0),
                "consumed_at": time.time() if now is None else float(now),
            }
            try:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                if created:
                    _fsync_dir(self.path.parent)
            except OSError as exc:
                raise LedgerError(f"ledger {self.path}: {exc}") from exc
            self._entries[decision_id] = entry["expires_at"]
        return None


_default = None


def default_ledger():
    """FileLedger at VERIFIX_LEDGER_PATH when set, else one ledger for this process."""
    global _default
    if _default is None:
        path = os.environ.get("VERIFIX_LEDGER_PATH")
        _default = FileLedger(path) if path else MemoryLedger()
    return _default
