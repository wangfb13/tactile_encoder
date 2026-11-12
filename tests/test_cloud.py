"""No-network tests of task-owned DLC admission, reconciliation, and locking."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))
import control


def response(value):
    return SimpleNamespace(body=SimpleNamespace(to_map=lambda: value))


class FakeCloud:
    def __init__(self):
        self.jobs = {}
        self.create_calls = 0
        self.timeout_before_accept = False
        self.timeout_after_accept = False

    def list_jobs(self, request):
        name = request.display_name
        return response({"Jobs": [dict(job) for job in self.jobs.values()
                                   if job["DisplayName"] == name]})

    def get_job(self, job_id, request):
        return response(dict(self.jobs[job_id]))

    def create_job_with_options(self, request, headers, runtime):
        self.create_calls += 1
        if self.timeout_before_accept:
            raise TimeoutError("simulated uncertain outcome")
        job_id = "fake-job-" + str(self.create_calls)
        self.jobs[job_id] = {"JobId": job_id, "DisplayName": request.display_name,
                             "Status": "Queuing", "Pods": []}
        if self.timeout_after_accept:
            raise TimeoutError("simulated lost create response")
        return response({"JobId": job_id, "RequestId": "fake-request"})

    def stop_job(self, job_id):
        self.jobs[job_id]["Status"] = "Stopping"
        return response({})


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cloud-unit-", dir=ROOT / "tests")
        self.cloud_root = Path(self.temporary.name)
        self.path_patch = patch.object(control, "CLOUD", self.cloud_root)
        self.path_patch.start()
        self.forbid_real_client = patch.object(control, "client", side_effect=AssertionError("Real API is forbidden in unit tests"))
        self.forbid_real_client.start()
        self.fake = FakeCloud()

    def tearDown(self):
        self.forbid_real_client.stop()
        self.path_patch.stop()
        self.temporary.cleanup()

    def test_queue_and_stopping_count_toward_two_job_limit(self):
        first = control.submit(self.fake, "one", "true")
        second = control.submit(self.fake, "two", "true")
        self.fake.jobs[second["JobId"]]["Status"] = "Stopping"
        with self.assertRaisesRegex(RuntimeError, "concurrency limit"):
            control.submit(self.fake, "three", "true")
        self.assertEqual(self.fake.create_calls, 2)
        rows = control.statuses(self.fake)
        self.assertEqual({row["Status"] for row in rows}, {"Queuing", "Stopping"})
        self.assertEqual(json.loads((self.cloud_root / "status.json").read_text())["active_jobs"], 2)
        self.fake.jobs[first["JobId"]]["Status"] = "Succeeded"
        control.submit(self.fake, "three", "true")
        self.assertEqual(self.fake.create_calls, 3)
        self.assertEqual(sum(x["Status"] not in control.TERMINAL for x in self.fake.jobs.values()), 2)

    def test_same_logical_submission_is_idempotent(self):
        first = control.submit(self.fake, "one", "true")
        again = control.submit(self.fake, "one", "true")
        self.assertEqual(first["JobId"], again["JobId"])
        self.assertTrue(again["already_submitted"])
        self.assertEqual(self.fake.create_calls, 1)

    def test_unknown_timeout_without_match_keeps_pending_and_never_reposts(self):
        self.fake.timeout_before_accept = True
        with self.assertRaises(TimeoutError):
            control.submit(self.fake, "uncertain", "true")
        pending = self.cloud_root / "jobs" / "uncertain" / "pending.json"
        self.assertTrue(pending.exists())
        self.fake.timeout_before_accept = False
        with self.assertRaisesRegex(RuntimeError, "Unresolved submission"):
            control.submit(self.fake, "uncertain", "true")
        self.assertTrue(pending.exists())
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(control.owned(), [])

    def test_lost_response_reconciles_existing_job_instead_of_reposting(self):
        self.fake.timeout_after_accept = True
        with self.assertRaises(TimeoutError):
            control.submit(self.fake, "accepted", "true")
        self.fake.timeout_after_accept = False
        result = control.submit(self.fake, "accepted", "true")
        self.assertTrue(result["already_submitted"])
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(len(control.owned()), 1)
        folder = self.cloud_root / "jobs" / "accepted"
        self.assertFalse((folder / "pending.json").exists())
        self.assertTrue((folder / "resolved.json").exists())
        self.assertEqual(json.loads((folder / "receipt.json").read_text())["JobId"], result["JobId"])

    def test_status_and_submit_both_wait_for_the_same_submission_lock(self):
        original = control._reconcile_unlocked
        for action in ("status", "submit"):
            entered = threading.Event()
            started = threading.Event()
            errors = []
            def spy(client):
                entered.set()
                return original(client)
            def worker():
                started.set()
                try:
                    if action == "status":
                        control.statuses(self.fake)
                    else:
                        control.submit(self.fake, "locked", "true")
                except Exception as error:
                    errors.append(error)
            with patch.object(control, "_reconcile_unlocked", spy):
                with (self.cloud_root / "submit.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    thread = threading.Thread(target=worker)
                    thread.start()
                    self.assertTrue(started.wait(1))
                    try:
                        self.assertFalse(entered.wait(.15), action + " bypassed the submission lock")
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                thread.join(3)
                self.assertFalse(thread.is_alive(), action + " deadlocked")
                self.assertEqual(errors, [])
                self.assertTrue(entered.is_set())

    def test_three_concurrent_submissions_admit_only_two(self):
        barrier = threading.Barrier(3)
        results, errors = [], []
        def worker(index):
            barrier.wait()
            try:
                results.append(control.submit(self.fake, "parallel_" + str(index), "true"))
            except Exception as error:
                errors.append(error)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(len(results), 2)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("concurrency limit", str(errors[0]))
        self.assertEqual(self.fake.create_calls, 2)
        self.assertEqual(len(control.owned()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
