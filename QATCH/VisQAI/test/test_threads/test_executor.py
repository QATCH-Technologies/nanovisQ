import unittest
import time
from typing import List
from src.threads.executor import Executor, ExecutionRecord


class DummyWorker:
    """A dummy worker object with various methods to exercise Executor."""

    def __init__(self):
        self.state = []

    def run(self, x, y=0):
        """Simple method that returns the sum and records state."""
        time.sleep(0.01)
        total = x + y
        self.state.append(total)
        return total

    def raise_error(self):
        """Method that always raises an exception."""
        raise RuntimeError("Intentional failure")

    not_callable = 42  # attribute that is not callable


class TestExecutionRecord(unittest.TestCase):
    def test_thread_property_before_start(self):
        """Accessing .thread before the thread is assigned should raise."""
        rec = ExecutionRecord(
            obj=None,
            method_name="run",
            args=(),
            kwargs={},
            thread_name="test-thread",
            callback=None
        )
        with self.assertRaises(RuntimeError) as cm:
            _ = rec.thread
        self.assertIn("Thread has not been created/started", str(cm.exception))

    def test_is_alive_and_join_before_start(self):
        """is_alive() should be False; join() before ._thread set should raise."""
        rec = ExecutionRecord(
            obj=None,
            method_name="run",
            args=(),
            kwargs={},
            thread_name="test-thread",
            callback=None
        )
        self.assertFalse(rec.is_alive())
        with self.assertRaises(RuntimeError) as cm:
            rec.join(timeout=0.01)
        self.assertIn("Cannot join before thread is started",
                      str(cm.exception))


class TestExecutor(unittest.TestCase):
    def setUp(self):
        self.executor = Executor()
        self.worker = DummyWorker()

    def test_run_invalid_method_name_raises(self):
        """If obj has no attribute method_name, run() should raise AttributeError."""
        with self.assertRaises(AttributeError):
            self.executor.run(self.worker, method_name="nonexistent_method")

    def test_run_non_callable_attribute_raises(self):
        """If the attribute exists but is not callable, run() should raise TypeError."""
        with self.assertRaises(TypeError):
            self.executor.run(self.worker, method_name="not_callable")

    def test_run_success_and_result(self):
        """A normal run should execute the method, set .result, and no exception."""
        rec = self.executor.run(self.worker, method_name="run",  x=3, y=4)
        rec.join(timeout=1.0)
        self.assertEqual(rec.result, 7)
        self.assertIsNone(rec.exception)
        self.assertIn(7, self.worker.state)

    def test_custom_thread_name_and_default_naming(self):
        """Verify that a provided thread_name is used, and default names increment."""
        rec1 = self.executor.run(
            self.worker, method_name="run",  x=1, y=2, thread_name="CustomName")
        self.assertEqual(rec1.thread.name, "CustomName")
        rec1.join(timeout=1.0)
        rec2 = self.executor.run(self.worker, method_name="run",  x=2, y=3)
        self.assertEqual(rec2.thread.name, "Thread-2")
        rec2.join(timeout=1.0)

    def test_callback_invoked(self):
        """Ensure that a callback is invoked exactly once with the record."""
        callback_calls: List[ExecutionRecord] = []

        def cb(record: ExecutionRecord):
            callback_calls.append(record)

        rec = self.executor.run(
            self.worker, method_name="run",  x=5, y=6, callback=cb)
        rec.join(timeout=1.0)

        self.assertEqual(len(callback_calls), 1)
        self.assertIs(callback_calls[0], rec)
        self.assertEqual(rec.result, 11)

    def test_callback_exception_swallowed(self):
        """If the callback itself raises, Executor should swallow it silently."""
        def bad_callback(record: ExecutionRecord):
            raise ValueError("Callback failure")
        rec = self.executor.run(
            self.worker, method_name="run",  x=2, y=2, callback=bad_callback)
        rec.join(timeout=1.0)
        self.assertEqual(rec.result, 4)
        self.assertIsNone(rec.exception)

    def test_exception_handling_in_task(self):
        """If the target method raises, record.exception should capture it."""
        rec = self.executor.run(self.worker, method_name="raise_error")
        rec.join(timeout=1.0)

        self.assertIsNone(rec.result)
        self.assertIsInstance(rec.exception, RuntimeError)
        self.assertIn("Intentional failure", str(rec.exception))

    def test_active_count_and_join_all(self):
        """Test active_count, join_all, and that join_all waits for tasks to complete."""
        class Sleeper:
            def long_run(self):
                time.sleep(0.1)
                return "done"

        sleeper = Sleeper()
        rec1 = self.executor.run(sleeper, method_name="long_run")
        rec2 = self.executor.run(sleeper, method_name="long_run")
        time.sleep(0.02)
        active_before = self.executor.active_count()
        self.assertGreaterEqual(active_before, 1)
        self.executor.join_all(timeout=1.0)
        active_after = self.executor.active_count()
        self.assertEqual(active_after, 0)
        results = self.executor.get_all_results()
        self.assertEqual(results.count("done"), 2)

    def test_get_all_exceptions_and_results_empty(self):
        """On a fresh Executor, get_all_results/exceptions should return empty lists."""
        self.assertEqual(self.executor.get_all_results(), [])
        self.assertEqual(self.executor.get_all_exceptions(), [])

    def test_get_all_exceptions_and_results_with_mixed_outcomes(self):
        """Verify get_all_results and get_all_exceptions reflect successes and failures."""
        rec_success = self.executor.run(
            self.worker, method_name="run",  x=1, y=1)
        rec_fail = self.executor.run(self.worker, method_name="raise_error")

        rec_success.join(timeout=1.0)
        rec_fail.join(timeout=1.0)

        results = self.executor.get_all_results()
        exceptions = self.executor.get_all_exceptions()

        self.assertEqual(len(results), 2)
        self.assertEqual(len(exceptions), 2)

        self.assertIn(2, results)
        self.assertIn(None, results)

        self.assertTrue(any(isinstance(exc, RuntimeError)
                        for exc in exceptions))
        self.assertIn(None, exceptions)

    def test_cleanup_finished(self):
        """cleanup_finished should remove records of tasks that are no longer alive."""
        class Short:
            def quick(self):
                return "quick"

        class Long:
            def slow(self):
                time.sleep(0.2)
                return "slow"

        short = Short()
        long = Long()

        rec_short = self.executor.run(short, method_name="quick")
        rec_long = self.executor.run(long, method_name="slow")

        rec_short.join(timeout=1.0)
        time.sleep(0.01)

        all_before = self.executor.get_task_records()
        self.assertEqual(len(all_before), 2)

        self.executor.cleanup_finished()
        remaining = self.executor.get_task_records()
        self.assertEqual(len(remaining), 1)
        self.assertIs(remaining[0], rec_long)

        rec_long.join(timeout=1.0)
        self.executor.cleanup_finished()
        self.assertEqual(self.executor.get_task_records(), [])

    def test_get_task_records_returns_copy(self):
        """Modifying the returned list of records should not affect internal state."""
        rec = self.executor.run(self.worker, method_name="run",  x=0, y=0)
        rec.join(timeout=1.0)

        records = self.executor.get_task_records()
        self.assertEqual(len(records), 1)

        records.pop()
