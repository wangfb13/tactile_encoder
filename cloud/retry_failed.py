"""Explicit, bounded recovery of one failed task-owned training job.

No background polling or automatic retries. The accepted original request is reused
byte-for-byte semantically, except for a new DisplayName. A logical experiment has
at most two cloud attempts. Unknown CreateJob outcomes remain pending until reconciled.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control as cloud


def request_hash(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


def folder_for(entry):
    folder = entry.get("folder", entry["logical"])
    if not isinstance(folder, str) or Path(folder).name != folder or folder in ("", ".", ".."):
        raise RuntimeError("Invalid attempt folder in task ledger")
    return cloud.CLOUD / "jobs" / folder


def require_resume_auto(command):
    tokens = shlex.split(command)
    return "--resume=auto" in tokens or any(
        a == "--resume" and b == "auto" for a, b in zip(tokens, tokens[1:]))


def retry_failed(c, logical, reason="Explicit recovery of failed task-owned training"):
    if not logical.replace("_", "").isalnum() or not logical.startswith(("screen_", "final_")):
        raise ValueError("Only a named screen_ or final_ training experiment can be recovered")
    with (cloud.CLOUD / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        known_ids = {entry["JobId"] for entry in cloud.owned()}
        # Reuses the same uncertain-outcome protection as normal submissions.
        entries = cloud._reconcile_unlocked(c)
        history = [entry for entry in entries if entry["logical"] == logical]
        if not history:
            raise RuntimeError("Unknown logical experiment; there is no owned attempt to recover")
        latest = history[-1]
        current = cloud.get_status(c, latest)
        if len(history) > 2:
            raise RuntimeError("Attempt ledger exceeds the maximum of two; manual investigation required")
        if len(history) == 2:
            if current["Status"] == "Failed":
                raise RuntimeError("Second attempt failed; maximum two attempts reached, no further automatic recovery")
            if current["Status"] == "Stopped":
                raise RuntimeError("Latest attempt was Stopped; user-stopped training is never restarted")
            # This path only reports an existing accepted attempt; it never creates a job.
            return {"already_submitted": True, "reconciled": latest["JobId"] not in known_ids,
                    "JobId": latest["JobId"], "logical": logical,
                    "attempt": 2, "status": current["Status"], "folder": latest.get("folder", logical)}
        if current["Status"] != "Failed":
            raise RuntimeError("Latest attempt must be Failed; refusing recovery of " + str(current["Status"]))
        if int(latest.get("attempt", 1)) != 1:
            raise RuntimeError("Inconsistent retry history; refusing to exceed two attempts")
        if int(latest["gpu"]) != 1:
            raise RuntimeError("Recovery is limited to a single-GPU training request")
        rows = [cloud.get_status(c, entry) for entry in entries]
        active = [row for row in rows if row["Status"] not in cloud.TERMINAL]
        if len(active) > 1 or sum(int(row["gpu"]) for row in active) + 1 > 2:
            raise RuntimeError("Task concurrency limit: at most two jobs/two GPUs including the recovery")

        previous_folder = folder_for(latest)
        original = json.loads((previous_folder / "request.json").read_text())
        if request_hash(original) != latest["request_sha256"]:
            raise RuntimeError("Original request changed since submission; recovery cannot change training parameters")
        if original.get("DisplayName") != latest["DisplayName"]:
            raise RuntimeError("Original request/ledger name mismatch")
        specs = original.get("JobSpecs", [])
        if (len(specs) != 1 or int(specs[0].get("PodCount", 0)) != 1 or
                int(specs[0].get("ResourceConfig", {}).get("GPU", 0)) != 1):
            raise RuntimeError("Original request is not one pod with exactly one GPU")
        if original.get("DryRun", False) or not require_resume_auto(original.get("UserCommand", "")):
            raise RuntimeError("Recovery requires the original non-dry-run training command with --resume auto")

        folder_name = logical + "__a2"
        folder = cloud.CLOUD / "jobs" / folder_name
        display_name = original["DisplayName"] + "_a2"
        matches = c.list_jobs(cloud.models.ListJobsRequest(
            workspace_id=original["WorkspaceId"], display_name=display_name,
            page_size=100, page_number=1)).body.to_map().get("Jobs", [])
        if any(row.get("DisplayName") == display_name for row in matches):
            raise RuntimeError("Retry cloud name already exists without a matching ledger; reconcile explicitly")
        if folder.exists():
            raise RuntimeError("Retry evidence folder already exists; inspect it instead of overwriting evidence")
        request = dict(original)
        request["DisplayName"] = display_name
        folder.mkdir(parents=True, exist_ok=False)
        cloud.save(folder / "request.json", request)
        pending = {"logical": logical, "folder": folder_name, "attempt": 2,
                   "DisplayName": display_name, "submitted_at": cloud.now(), "gpu": 1,
                   "request_sha256": request_hash(request), "retry_of": latest["JobId"],
                   "original_request_sha256": latest["request_sha256"],
                   "recovery_reason": str(reason)}
        cloud.save(folder / "pending.json", pending)
        result = c.create_job_with_options(
            cloud.models.CreateJobRequest().from_map(request), {},
            cloud.RuntimeOptions(autoretry=False, max_attempts=1)).body.to_map()
        entry = {**pending, "JobId": result["JobId"]}
        entries.append(entry)
        cloud.save(cloud.CLOUD / "owned_jobs.json", entries)
        cloud.save(folder / "receipt.json",
                   {"JobId": entry["JobId"], "RequestId": result.get("RequestId"),
                    "retry_of": latest["JobId"], "attempt": 2})
        (folder / "pending.json").rename(folder / "resolved.json")
        return entry


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--logical", required=True)
    parser.add_argument("--reason", default="Explicit recovery of failed task-owned training")
    args = parser.parse_args()
    try:
        result = retry_failed(cloud.client(), args.logical, args.reason)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result = {"error_type": type(exc).__name__, "code": getattr(exc, "code", None),
                  "status_code": getattr(exc, "status_code", None)}
        if isinstance(exc, (RuntimeError, ValueError, FileNotFoundError)):
            result["reason"] = str(exc)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
