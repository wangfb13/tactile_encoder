"""Safe recovery is tested entirely against a fake DLC client; no real API calls."""
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
import retry_failed


def response(value):
    return SimpleNamespace(body=SimpleNamespace(to_map=lambda: value))


class FakeCloud:
    def __init__(self):
        self.jobs = {}
        self.requests = []
        self.create_calls = 0
        self.timeout = None

    def get_job(self, job_id, request):
        return response(dict(self.jobs[job_id]))

    def list_jobs(self, request):
        return response({"Jobs": [dict(value) for value in self.jobs.values()
                                   if value["DisplayName"] == request.display_name]})

    def create_job_with_options(self, request, headers, runtime):
        self.create_calls += 1
        self.requests.append(request.to_map())
        if self.timeout == "before":
            raise TimeoutError("simulated unknown submission")
        job_id = "retry-job-" + str(self.create_calls)
        self.jobs[job_id] = {"JobId": job_id, "DisplayName": request.display_name,
                            "Status": "Queuing", "Pods": []}
        if self.timeout == "after":
            raise TimeoutError("simulated lost create response")
        return response({"JobId": job_id, "RequestId": "fake-request"})


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="retry-unit-", dir=ROOT / "tests")
        self.directory = Path(self.temporary.name)
        self.cloud_patch = patch.object(control, "CLOUD", self.directory)
        self.cloud_patch.start()
        self.forbid_real_client = patch.object(control, "client", side_effect=AssertionError("Real cloud API forbidden"))
        self.forbid_real_client.start()
        self.fake = FakeCloud()
        self.original = self.add_owned("screen_l010_s0", "Failed")

    def tearDown(self):
        self.forbid_real_client.stop()
        self.cloud_patch.stop()
        self.temporary.cleanup()

    def add_owned(self, logical, status, gpu=1, command=None):
        command = command or "set -euo pipefail\npython train.py --out runs/" + logical + " --resume auto\n"
        request = control.request(logical, command, gpu=gpu, minutes=360)
        entries = control.owned()
        job_id = "original-job-" + str(len(entries) + 1)
        entry = {"logical": logical, "DisplayName": request["DisplayName"],
                 "JobId": job_id, "gpu": gpu, "submitted_at": control.now(),
                 "request_sha256": retry_failed.request_hash(request)}
        folder = self.directory / "jobs" / logical
        control.save(folder / "request.json", request)
        control.save(folder / "receipt.json", {"JobId": job_id})
        entries.append(entry)
        control.save(self.directory / "owned_jobs.json", entries)
        self.fake.jobs[job_id] = {"JobId": job_id, "DisplayName": request["DisplayName"],
                                 "Status": status, "Pods": []}
        return entry

    def test_failed_job_reuses_request_and_keeps_original_evidence(self):
        folder = self.directory / "jobs" / self.original["logical"]
        original_bytes = (folder / "request.json").read_bytes()
        original_request = json.loads(original_bytes)
        result = retry_failed.retry_failed(self.fake, self.original["logical"], "transient node failure")
        request = self.fake.requests[-1]
        expected_wire = control.models.CreateJobRequest().from_map(original_request).to_map()
        self.assertEqual({k: v for k, v in request.items() if k != "DisplayName"},
                         {k: v for k, v in expected_wire.items() if k != "DisplayName"})
        stored_retry = json.loads((self.directory / "jobs" / result["folder"] / "request.json").read_text())
        self.assertEqual({k: v for k, v in stored_retry.items() if k != "DisplayName"},
                         {k: v for k, v in original_request.items() if k != "DisplayName"})
        self.assertEqual(request["DisplayName"], self.original["DisplayName"] + "_a2")
        self.assertEqual(result["folder"], self.original["logical"] + "__a2")
        self.assertEqual(result["logical"], self.original["logical"])
        self.assertEqual(result["retry_of"], self.original["JobId"])
        self.assertEqual(result["attempt"], 2)
        self.assertEqual((folder / "request.json").read_bytes(), original_bytes)
        retry_folder = self.directory / "jobs" / result["folder"]
        self.assertTrue((retry_folder / "receipt.json").exists())
        self.assertTrue((retry_folder / "resolved.json").exists())
        self.assertFalse((retry_folder / "pending.json").exists())
        self.assertEqual(len(control.owned()), 2)

    def test_stopped_or_nonfailed_original_is_never_restarted(self):
        for status in ("Stopped", "Running", "Queuing", "Stopping", "Succeeded"):
            self.fake.jobs[self.original["JobId"]]["Status"] = status
            with self.assertRaisesRegex(RuntimeError, "Latest attempt must be Failed"):
                retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(self.fake.create_calls, 0)

    def test_two_active_jobs_including_stopping_block_recovery(self):
        self.add_owned("screen_other_a", "Queuing")
        self.add_owned("screen_other_b", "Stopping")
        with self.assertRaisesRegex(RuntimeError, "concurrency limit"):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(self.fake.create_calls, 0)

    def test_one_other_active_job_allows_exactly_one_recovery(self):
        self.add_owned("screen_other", "Running")
        retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(sum(x["Status"] not in control.TERMINAL for x in self.fake.jobs.values()), 2)

    def test_second_failure_exhausts_attempt_budget(self):
        result = retry_failed.retry_failed(self.fake, self.original["logical"])
        self.fake.jobs[result["JobId"]]["Status"] = "Failed"
        with self.assertRaisesRegex(RuntimeError, "Second attempt failed"):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(self.fake.create_calls, 1)

    def test_existing_active_retry_is_reported_without_duplicate_submission(self):
        result = retry_failed.retry_failed(self.fake, self.original["logical"])
        again = retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(again["JobId"], result["JobId"])
        self.assertTrue(again["already_submitted"])
        self.assertEqual(self.fake.create_calls, 1)

    def test_unknown_timeout_keeps_pending_and_blocks_blind_retry(self):
        self.fake.timeout = "before"
        with self.assertRaises(TimeoutError):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        pending = self.directory / "jobs" / (self.original["logical"] + "__a2") / "pending.json"
        self.assertTrue(pending.exists())
        self.fake.timeout = None
        with self.assertRaisesRegex(RuntimeError, "Unresolved submission"):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertTrue(pending.exists())
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(len(control.owned()), 1)

    def test_lost_response_is_reconciled_without_reposting(self):
        self.fake.timeout = "after"
        with self.assertRaises(TimeoutError):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        self.fake.timeout = None
        result = retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertTrue(result["already_submitted"])
        self.assertTrue(result["reconciled"])
        self.assertEqual(self.fake.create_calls, 1)
        self.assertEqual(len(control.owned()), 2)
        pending = self.directory / "jobs" / (self.original["logical"] + "__a2") / "pending.json"
        self.assertFalse(pending.exists())

    def test_original_request_tampering_is_rejected(self):
        path = self.directory / "jobs" / self.original["logical"] / "request.json"
        request = json.loads(path.read_text())
        request["UserCommand"] += "\necho changed\n"
        control.save(path, request)
        with self.assertRaisesRegex(RuntimeError, "Original request changed"):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(self.fake.create_calls, 0)

    def test_nonresume_command_cannot_be_recovered(self):
        path = self.directory / "jobs" / self.original["logical"] / "request.json"
        request = json.loads(path.read_text())
        request["UserCommand"] = "python train.py"
        control.save(path, request)
        entries = control.owned()
        entries[0]["request_sha256"] = retry_failed.request_hash(request)
        control.save(self.directory / "owned_jobs.json", entries)
        with self.assertRaisesRegex(RuntimeError, "--resume auto"):
            retry_failed.retry_failed(self.fake, self.original["logical"])
        self.assertEqual(self.fake.create_calls, 0)

    def test_recovery_waits_for_same_submission_lock(self):
        started, finished = threading.Event(), threading.Event()
        errors = []
        def worker():
            started.set()
            try:
                retry_failed.retry_failed(self.fake, self.original["logical"])
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()
        with (self.directory / "submit.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(started.wait(1))
            try:
                self.assertFalse(finished.wait(.15))
                self.assertEqual(self.fake.create_calls, 0)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.fake.create_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
