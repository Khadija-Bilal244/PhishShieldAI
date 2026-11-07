#!/usr/bin/env bash
# One-time project setup: Git + DVC + DagsHub remote. Reads credentials from .env
#   bash scripts/bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { cp .env.example .env; echo ">> Created .env - fill in your DagsHub details, then re-run."; exit 1; }
set -a; source .env; set +a

[ -d .git ] || { git init -b main; echo ">> git initialised"; }
[ -d .dvc ] || { dvc init; echo ">> dvc initialised"; }

if [ -n "${DAGSHUB_USERNAME:-}" ] && [ "${DAGSHUB_USERNAME}" != "your-dagshub-username" ]; then
  REMOTE="https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO}.dvc"
  dvc remote add -d -f origin "$REMOTE"
  # credentials go to .dvc/config.local (never committed)
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USERNAME"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
  echo ">> DVC remote 'origin' -> $REMOTE"
  git remote get-url origin >/dev/null 2>&1 || echo ">> TIP: git remote add origin https://github.com/<you>/${DAGSHUB_REPO}.git"
else
  echo ">> DagsHub not configured in .env - skipping DVC remote (local only)."
fi

git add .dvc .dvcignore .gitignore 2>/dev/null || true
echo ">> Done. Next: dvc repro   (see docs/SETUP.md)"
