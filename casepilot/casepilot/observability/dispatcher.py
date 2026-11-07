"""One failure-isolated dispatch path. Cloud I/O runs on a bounded daemon worker."""

import atexit
import queue
import threading
import time

from casepilot.observability.events import EVENTS, metadata


class Dispatcher:
    def __init__(self, *, otlp=None, langsmith=None, queue_size=256):
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self._otlp = otlp
        self._langsmith = langsmith
        self._queue = queue.Queue(maxsize=queue_size)
        self._worker = None
        self._closed = False
        self._lock = threading.Lock()
        self._counts = dict(otlp_failures=0, langsmith_failures=0, langsmith_dropped=0)

    @property
    def diagnostics(self):
        with self._lock:
            return dict(self._counts)

    def _count(self, name):
        with self._lock:
            self._counts[name] = min(self._counts[name] + 1, 2**63 - 1)

    def _call(self, name, exporter, event, record):
        try:
            return exporter(event, metadata(record))
        except Exception:
            self._count(name + "_failures")
            return None

    def dispatch(self, event, record):
        from casepilot.observability.exporters import langsmith, otlp

        if event not in EVENTS:
            raise ValueError("Unsupported trace event")
        safe = metadata(record)
        with self._lock:
            if self._closed:
                return None
        trace_id = self._call("otlp", self._otlp or otlp.export_event, event, safe)
        if self._langsmith is not None or langsmith.enabled():
            with self._lock:
                if self._closed:
                    return trace_id
                if self._worker is None:
                    self._worker = threading.Thread(
                        target=self._run, daemon=True, name="casepilot-telemetry"
                    )
                    self._worker.start()
                try:
                    self._queue.put_nowait((event, metadata(safe)))
                except queue.Full:
                    self._counts["langsmith_dropped"] = min(
                        self._counts["langsmith_dropped"] + 1, 2**63 - 1
                    )
        return trace_id

    def _run(self):
        from casepilot.observability.exporters import langsmith

        try:
            while True:
                try:
                    event, record = self._queue.get(timeout=0.05)
                except queue.Empty:
                    with self._lock:
                        if self._closed:
                            break
                    continue
                try:
                    self._call(
                        "langsmith", self._langsmith or langsmith.export_event, event, record
                    )
                finally:
                    self._queue.task_done()
        finally:
            if self._langsmith is None:
                try:
                    langsmith.close()
                except Exception:
                    self._count("langsmith_failures")

    def flush(self, timeout=5.0):
        """Wait at most timeout seconds for queued cloud exports; false means pending work."""
        deadline = time.monotonic() + max(0, timeout)
        with self._queue.all_tasks_done:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue.all_tasks_done.wait(remaining)
        return True

    def close(self, timeout=5.0):
        """Stop accepting work and drain within timeout; a stalled exporter stays daemonized."""
        deadline = time.monotonic() + max(0, timeout)
        with self._lock:
            self._closed = True
            worker = self._worker
        if worker is not None:
            worker.join(max(0, deadline - time.monotonic()))
            return not worker.is_alive()
        return True


_default = Dispatcher()
atexit.register(_default.close, 0.5)


def get_dispatcher():
    return _default


def flush(timeout=5.0):
    """Drain the shared cloud queue within the supplied seconds."""
    return _default.flush(timeout)


def close(timeout=5.0):
    """Bounded application shutdown; individual customer sessions must not call this."""
    return _default.close(timeout)


def diagnostics():
    """Return only fixed exporter names and saturated counts, never exceptions or payloads."""
    return _default.diagnostics
