# 06 · Free cloud deployment (Render) and Kubernetes

Everything here is free and needs **no credit card**.

> Why not Hugging Face Spaces? Since July 2026, Docker and Gradio Spaces need a paid PRO plan. Only static Spaces are still free.

## Render (public URL, free web service)

The free web service has 512 MB RAM and 750 hours per month. It sleeps after 15 minutes idle and takes about 1 minute to wake.
The API uses about 330 MB at peak, measured with continuous predictions and explanations, so it fits.

CD pushes a tested image, with the champion model built in, to the GitHub Container Registry. Render runs that image.

1. **Make the image public (once):** GitHub → your profile → **Packages** → `phishshield-api` → **Package settings** →
   **Change visibility → Public**. Run **Actions → CD** once first so the package exists.
2. **Create the service:** sign up at https://render.com with GitHub (no card). Then either:
   - **New → Blueprint**, pick your repo, and Render reads `render.yaml`; or
   - **New → Web Service → Existing image** → `ghcr.io/khadija-bilal244/phishshieldai/phishshield-api:latest`,
     instance type **Free**, and add the environment variable `PORT=8000`.
3. **Auto-deploy from CD:** Render service → **Settings → Deploy Hook** → copy the URL. In GitHub repo → **Settings → Secrets → Actions**, add:
   - `RENDER_DEPLOY_HOOK` = the hook URL
   - `RENDER_URL` = `https://phishshield-api-xxxx.onrender.com` (shown on the service page)
4. Every successful CD run now pushes the image, redeploys Render, and smoke-tests the public URL.

**Logs and monitoring:**
- The Render dashboard → **Logs** and **Metrics** tabs give live container logs, CPU and memory.
- Prometheus can also scrape the live service. Add this job to `monitoring/prometheus.yml` and run `docker compose up -d`:

```yaml
  - job_name: phishshield-render
    scheme: https
    metrics_path: /metrics
    static_configs:
      - targets: ["phishshield-api-xxxx.onrender.com"]
```

- Drift reports come from the daily **Drift Monitor** workflow (Actions → artifacts + issues).

Open the URL a minute before a demo so the service is awake.

## Kubernetes (local, free)

Use Docker Desktop (Settings → Kubernetes → Enable), or `minikube start` + `minikube addons enable metrics-server`.

```bash
kubectl apply -f k8s/
kubectl -n phishshield get pods,svc,hpa
kubectl -n phishshield port-forward svc/phishshield-api 8000:80
```

Things to demonstrate:

```bash
kubectl -n phishshield rollout status deploy/phishshield-api
kubectl -n phishshield rollout undo deploy/phishshield-api
kubectl -n phishshield delete pod -l app=phishshield-api --wait=false
python scripts/load_test.py --n 3000 & kubectl -n phishshield get hpa -w
```
