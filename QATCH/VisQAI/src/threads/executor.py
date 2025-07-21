from threading import Thread, Lock
from typing import Any, Callable, List, Optional


class ExecutionRecord:
    def __init__(
        self,
        obj: Any,
        method_name: str,
        args: tuple,
        kwargs: dict,
        thread_name: str,
        callback: Optional[Callable[["ExecutionRecord"], None]],
    ) -> None:
        self.obj = obj
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.thread_name = thread_name
        self.callback = callback

        self.result: Any = None
        self.exception: Optional[Exception] = None
        self._thread: Optional[Thread] = None

    @property
    def thread(self) -> Thread:
        if self._thread is None:
            raise RuntimeError("Thread has not been created/started yet.")
        return self._thread

    def is_alive(self) -> bool:
        return self._thread.is_alive() if self._thread is not None else False

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        else:
            raise RuntimeError("Cannot join before thread is started.")


class Executor:
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: List[ExecutionRecord] = []

    def run(
        self,
        obj: Any,
        method_name: str = "run",
        *args,
        thread_name: Optional[str] = None,
        callback: Optional[Callable[[ExecutionRecord], None]] = None,
        **kwargs
    ) -> ExecutionRecord:
        if not hasattr(obj, method_name):
            raise AttributeError(f"{obj!r} has no attribute '{method_name}()'")

        method = getattr(obj, method_name)
        if not callable(method):
            raise TypeError(
                f"Attribute '{method_name}' of {obj!r} is not callable")

        default_name = f"Thread-{len(self._tasks) + 1}"
        final_thread_name = thread_name or default_name

        record = ExecutionRecord(
            obj=obj,
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            thread_name=final_thread_name,
            callback=callback,
        )

        def _wrapper() -> None:
            try:
                result = method(*args, **kwargs)
                record.result = result
            except Exception as e:
                record.exception = e
            finally:
                if record.callback:
                    try:
                        record.callback(record)
                    except Exception:
                        pass

        thread = Thread(target=_wrapper, name=final_thread_name, daemon=False)
        record._thread = thread

        with self._lock:
            self._tasks.append(record)

        thread.start()
        return record

    def join_all(self, timeout: Optional[float] = None) -> None:
        with self._lock:
            tasks_snapshot = list(self._tasks)

        for rec in tasks_snapshot:
            rec.join(timeout=timeout)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for rec in self._tasks if rec.is_alive())

    def task_count(self) -> int:
        with self._lock:
            return len(self._tasks)

    def get_all_results(self) -> List[Any]:
        with self._lock:
            return [rec.result for rec in self._tasks]

    def get_all_exceptions(self) -> List[Optional[Exception]]:
        with self._lock:
            return [rec.exception for rec in self._tasks]

    def cleanup_finished(self) -> None:
        with self._lock:
            self._tasks = [rec for rec in self._tasks if rec.is_alive()]

    def get_task_records(self) -> List[ExecutionRecord]:
        with self._lock:
            return list(self._tasks)
