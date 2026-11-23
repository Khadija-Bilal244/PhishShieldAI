"""Airflow DAGs for PhishShield AI (Airflow 3.x, with Airflow 2.x fallback imports)."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

try:
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.providers.standard.operators.empty import EmptyOperator
    from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
    from airflow.sdk import DAG
except ImportError:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator, PythonOperator

ROOT = Path(os.environ.get("PHISHSHIELD_ROOT", "/opt/phishshield"))
PY = f"cd {ROOT} && PYTHONPATH={ROOT}/src PHISHSHIELD_ROOT={ROOT} python -m"
API_URL = os.environ.get("PHISHSHIELD_API_URL", "http://phishshield-api:8000")
ENV = {**os.environ, "PYTHONPATH": f"{ROOT}/src", "PHISHSHIELD_ROOT": str(ROOT)}

default_args = {"owner": "phishshield", "retries": 1, "retry_delay": timedelta(minutes=2)}


def collect_batch(**context) -> str:
    """Returns the CSV the rest of the run works on (pushed to XCom)."""
    feedback = ROOT / "data" / "production" / "feedback.csv"
    if feedback.exists() and sum(1 for _ in open(feedback)) > 200:
        print("Using labelled user feedback")
        return str(feedback)
    r = subprocess.run(["python", "-m", "phishshield.data.collect_live"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    latest = ROOT / "data" / "live" / "latest.json"
    if r.returncode == 0 and latest.exists():
        f = ROOT / "data" / "live" / json.loads(latest.read_text())["file"]
        print(f"Using live batch {f.name}")
        return str(f)
    batches = sorted((ROOT / "data" / "stream").glob("batch_*.csv"))
    if not batches:
        raise FileNotFoundError("No live data and no stream batches - run `dvc repro` first")
    day = int(context["logical_date"].timestamp() // 86400) if context.get("logical_date") else 0
    print(f"Live collection failed -> simulated stream batch {batches[day % len(batches)].name}")
    return str(batches[day % len(batches)])


def branch_on_risk() -> str:
    decision = json.loads((ROOT / "reports" / "drift" / "latest.json").read_text())["decision"]
    print(f"risk={decision['risk']:.3f} action={decision['action']}")
    for reason in decision["reasons"]:
        print("  -", reason)
    return "no_action" if decision["action"] == "none" else "update_model"


def reload_api() -> None:
    import urllib.request

    req = urllib.request.Request(f"{API_URL}/admin/reload", method="POST",
                                 headers={"X-Admin-Token": os.environ.get("ADMIN_TOKEN", "change-me")})
    try:
        print(urllib.request.urlopen(req, timeout=10).read().decode())
    except Exception as e:
        print(f"API reload skipped: {e}")


def summary() -> None:
    for name in ("latest.json", "latest_update.json"):
        p = ROOT / "reports" / "drift" / name
        print(name, "->", p.read_text()[:3000] if p.exists() else "not produced in this run")


with DAG(
    dag_id="phishshield_adaptive_loop",
    description="Daily: collect data -> drift check -> tiered risk-based update -> promote/reject -> reload API",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["phishshield", "drift", "mlops"],
) as adaptive_dag:
    t_collect = PythonOperator(task_id="collect_batch", python_callable=collect_batch)
    t_drift = BashOperator(
        task_id="drift_check",
        bash_command=f"{PY} phishshield.drift.monitor --batch {{{{ ti.xcom_pull(task_ids='collect_batch') }}}}",
    )
    t_branch = BranchPythonOperator(task_id="branch_on_risk", python_callable=branch_on_risk)
    t_none = EmptyOperator(task_id="no_action")
    t_update = BashOperator(
        task_id="update_model",
        bash_command=f"{PY} phishshield.pipeline.update --batch {{{{ ti.xcom_pull(task_ids='collect_batch') }}}}",
        execution_timeout=timedelta(hours=3),
    )
    t_reload = PythonOperator(task_id="reload_api", python_callable=reload_api)
    t_version = BashOperator(
        task_id="version_registry",
        bash_command=f"cd {ROOT} && dvc add models/registry && (dvc push models/registry.dvc || echo 'no remote')",
    )
    t_summary = PythonOperator(task_id="summary", python_callable=summary, trigger_rule="none_failed_min_one_success")

    t_collect >> t_drift >> t_branch >> [t_none, t_update]
    t_update >> t_reload >> t_version >> t_summary
    t_none >> t_summary


with DAG(
    dag_id="phishshield_training_pipeline",
    description="Reproducible training pipeline (DVC) with remote storage on DagsHub",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["phishshield", "training"],
) as train_dag:
    pull = BashOperator(task_id="dvc_pull", bash_command=f"cd {ROOT} && (dvc pull || echo 'nothing to pull')")
    repro = BashOperator(task_id="dvc_repro", bash_command=f"cd {ROOT} && PYTHONPATH={ROOT}/src dvc repro",
                         execution_timeout=timedelta(hours=6))
    push = BashOperator(task_id="dvc_push", bash_command=f"cd {ROOT} && (dvc push || echo 'no DVC remote configured')")
    pull >> repro >> push
