# 04 · Apache Airflow

Two DAGs live in `airflow/dags/phishshield_dag.py`.

### `phishshield_adaptive_loop` (daily)

```
collect_batch ─► drift_check ─► branch_on_risk ─┬─► no_action ──────────────────────────────────────┐
                                                └─► update_model ─► reload_api ─► version_registry ──┴─► summary
```

- **collect_batch** uses labelled feedback if there are more than 200 rows, otherwise **today's live OpenPhish + Tranco data**, otherwise the next stream batch.
- **drift_check** runs the drift monitor. **branch_on_risk** follows the policy's decision.
- **update_model** runs the tiered update and the promotion gate. **reload_api** hot-swaps the API.
- **version_registry** runs `dvc add models/registry && dvc push`.

### `phishshield_training_pipeline` (weekly)

`dvc_pull ─► dvc_repro ─► dvc_push`

## Run it

```bash
docker compose up -d --build                                  # API stack first (Airflow joins its network)
docker compose -f docker-compose.airflow.yml up -d --build    # first build ~10 min
docker compose -f docker-compose.airflow.yml logs airflow | grep -i password
```

- Linux only: `echo "AIRFLOW_UID=$(id -u)" >> .env` before starting.
- Your `.env` (DagsHub MLflow variables) is passed to Airflow, so update runs are logged to DagsHub.

Open http://localhost:8080 and log in as `admin` with the password from the logs. Turn on **phishshield_adaptive_loop**,
click **Trigger**, then open the run's **Graph**. One branch turns green and the other is skipped. Click a task, then **Logs**,
to see the risk score and the reasons.

`PHISHSHIELD_TINY` defaults to 0 (real DistilBERT). On a laptop CPU, a `full_retrain` tier will take a long time. For a demo, set
`PHISHSHIELD_TINY=1` in `.env`, but that writes tiny-model candidates into the registry. Use that only for demos and roll back afterwards.

## Viva concepts

| Term | Here |
|---|---|
| DAG / task / operator | `with DAG(...)`, `BashOperator`, `PythonOperator`, `BranchPythonOperator`, `EmptyOperator` |
| XCom | `collect_batch` returns the batch path, and later tasks read it with `ti.xcom_pull` |
| Branching + trigger rule | the skipped branch doesn't block `summary` (`none_failed_min_one_success`) |
| Retries / schedule / catchup | 1 retry after 2 min; `@daily`; `catchup=False` |
