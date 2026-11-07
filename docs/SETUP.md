# Setup guide: from zip to submission

Follow the steps in order. Commands are shown for **Windows PowerShell**, with Linux/macOS versions where they differ.

---

## 1. Local install (once per person)

```powershell
cd phishshield-ai
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
pip install -e .
pytest                              # all tests should pass
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. GitHub repository (group leader)

1. Create an **empty public** repo `phishshield-ai` on GitHub (no README).
2. Then:

```powershell
git init -b main
dvc init
git add .
git commit -m "Initial project structure"
git remote add origin https://github.com/<user>/phishshield-ai.git
git push -u origin main
```

3. **Settings → Collaborators → Add people**: add every group member.

## 3. DagsHub (DVC remote + MLflow server)

1. Sign up at https://dagshub.com **with GitHub**.
2. **Create → New Repository → Connect a repository → GitHub** and pick `phishshield-ai`. DagsHub mirrors it.
3. **Settings → Collaborators** on DagsHub: add the group members.
4. Token: avatar → **Your Settings → Tokens** and copy it.
5. Fill in `.env`:

```powershell
copy .env.example .env               # Linux/macOS: cp .env.example .env
notepad .env                         # username, repo, token
```

## 4. Connect DVC to DagsHub

```powershell
dvc remote add -d origin https://dagshub.com/<user>/phishshield-ai.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
git add .dvc/config
git commit -m "Add DagsHub DVC remote"
git push
```

`--local` keeps the token in `.dvc/config.local`, which is git-ignored.
You can also run `powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1`, which does steps 4 and 5 from `.env`.

## 5. Run the full pipeline (real DistilBERT, on a GPU)

DistilBERT is too slow on a laptop CPU, so use **Google Colab**:

1. https://colab.research.google.com → **File → Upload notebook** → `notebooks/colab_run_pipeline.ipynb`
2. **Runtime → Change runtime type → T4 GPU**
3. Set `GITHUB_USER` in the first cell and run every cell. You'll be asked for the DagsHub token and a GitHub token
   (GitHub → Settings → Developer settings → Personal access tokens, with `repo` scope).

The notebook runs `dvc repro`, which logs everything to DagsHub MLflow and registers `phishshield-hybrid` in **Staging**.
It then runs `dvc push` (data and models to DagsHub) and `git push` (`dvc.lock` and reports to GitHub).

**Laptop alternative (CPU, slower):** in `params.yaml` set `hybrid.max_train_samples: 8000` and `hybrid.epochs: 2`.
Then run `dvc repro`, followed by `dvc push`, `git add dvc.lock reports`, `git commit`, `git push`. Expect about 30–60 minutes.

**Quick CPU smoke test (not for submission):** `$env:PHISHSHIELD_TINY="1"; dvc repro`. This uses a small offline
transformer. Afterwards run `dvc repro -f train_hybrid` without the variable, because DVC does not track env vars.

## 6. Check that everything worked

| Where | What you should see |
|---|---|
| DagsHub repo → **Data Pipeline** tab | the 8-stage DAG |
| DagsHub repo → files `data/`, `models/` | the DVC badge (stored on DagsHub) |
| DagsHub → **Experiments** (or *Remote → Experiments → Go to MLflow UI*) | 5 runs: 3 baselines, `hybrid-fusion`, `evaluation-test-set` |
| MLflow UI → **Models** | `phishshield-hybrid` version 1 in **Staging**, and `phishshield-baseline-xgboost` |
| Terminal: `dvc repro` | `Data and pipelines are up to date` (reproducible) |

## 7. Team workflow (every member needs commits)

Each member clones the repo, does steps 1 and 4 (with their own DagsHub token), and works on a branch:

```powershell
git clone https://github.com/<leader>/phishshield-ai.git
cd phishshield-ai
dvc pull
git checkout -b <name>/<task>
# ... make the change ...
git add <files>
git commit -m "Clear message"
git push -u origin <name>/<task>
```

Open a **Pull Request** on GitHub, let CI go green, and merge it. A suggested split so everyone has meaningful commits:

| Member | Owns | Example commits |
|---|---|---|
| A | Data: `data/ingest.py`, `validate.py`, `split.py` | add a validation check, tune `validation` params |
| B | Baselines + features: `features/`, `models/baselines.py` | add a feature, tune XGBoost params, write tests |
| C | Main model + MLflow: `models/hybrid.py`, `training/`, `registry/` | GPU run, MLflow registration, README results |

Use your **own** Git identity on your own laptop (`git config --global user.name/user.email` with the email of your
GitHub account) so the commits count for you.

## 8. Screenshots for the PDF

1. **MLflow runs** (at least 2): MLflow UI → experiment `phishshield-ai`. Take one screenshot of the runs list, and one of
   *Compare* with 2–3 runs selected (or the `hybrid-fusion` run page showing metrics and the training curve).
2. **Registered model in Staging**: MLflow UI → **Models → phishshield-hybrid**, showing *Stage: Staging*.
3. **DVC pipeline diagram**: run `dvc dag` in the terminal and take a screenshot, or use the DagsHub *Data Pipeline* tab.
   `dvc dag --md` gives a Mermaid version.

Insert them into `PhishShield_Milestone1_Report.docx`, add your two links, then **File → Save As → PDF**.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named phishshield` | `pip install -e .` |
| `ingest` download fails | Download the zip from https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset and save it as `data/raw/phiusiil.zip` |
| Warning *"logging to LOCAL"* | `.env` not filled in, so MLflow isn't pointing to DagsHub |
| `dvc push` 401/403 | Re-run the three `dvc remote modify origin --local` lines with a valid token |
| CUDA out of memory | `hybrid.batch_size: 16` in `params.yaml` |
| Stage transition error | Use the `mlflow<3` version in `requirements.txt`. The code also sets a `staging` alias as a fallback |
