# Deploying Property Guide (GitHub + Render)

Steps for deploying your own copy from a GitHub repository to Render.

**Time:** about 20 minutes, plus 5–10 minutes for Render's first build.
**You need:** a GitHub account, a Render account (https://render.com), and an OpenRouter API key.

---

## 1. Create the GitHub repository

**Option A: GitHub CLI** (`gh auth login` first):
```bash
gh repo create realestate-chatbot --public --source . --remote origin --push
```

**Option B: website.** On https://github.com/new, create an **empty** public repo named `realestate-chatbot` (no README/licence), then:
```bash
git remote add origin https://github.com/<your-user>/realestate-chatbot.git
git push -u origin main
```

**Check that no secrets went up:**
```bash
git ls-files | grep -i env                          # must print only: .env.example
git log -p | grep -E "sk-or-[A-Za-z0-9]" || echo "no keys in history"
```

The **CI** workflow (`.github/workflows/ci.yml`: ruff + 79 tests, no network needed) runs on the first push. Check the *Actions* tab is green.

## 2. Deploy on Render (Docker, free plan)

1. Render dashboard → **New → Blueprint** → connect GitHub → select `realestate-chatbot`. Render reads `render.yaml`: a Docker web service on the free plan, with health check `/api/health`.
2. When Render asks for the variables marked `sync: false`:
   - **`OPENROUTER_API_KEY`** → paste your key here. This is the **only** place the key goes. Render stores it as a secret environment variable; it is **never** written to the repo or the Docker image.
   - `APP_PUBLIC_URL` → leave empty for now (set in section 3).
   - `REPO_URL` → `https://github.com/<your-user>/realestate-chatbot` (shown as "Source code" in the UI footer).
3. **Apply.** The first build takes 5–10 minutes: it installs dependencies, downloads two small ONNX models, and builds the search index inside the image.

The blueprint already sets the free-tier tuning: `USE_RERANKER=false` and `ONNX_THREADS=1` (512 MB RAM / 0.1 CPU), `PREWARM_CACHE=true`, `PORT=7860`. This configuration was tested locally under exactly those limits: about 293 MB RAM, 10/10 acceptance questions.

*Alternative without the blueprint:* **New → Web Service** → your repo → Language **Docker** → Instance type **Free** → Health Check Path `/api/health` → add the environment variables above plus `USE_RERANKER=false`, `ONNX_THREADS=1`, `PREWARM_CACHE=true`, `PORT=7860`.

## 3. After the first deploy

1. Copy the service URL, e.g. `https://realestate-chatbot-xxxx.onrender.com`.
2. Render → service → **Environment** → set `APP_PUBLIC_URL` to that URL (it's sent to OpenRouter as the referer). Save; Render redeploys.
3. Verify:
   ```bash
   curl https://<your-service>.onrender.com/api/health     # "status":"ok", "listings":3901, "llm_configured":true
   python eval/acceptance.py --base-url https://<your-service>.onrender.com   # expect 10/10
   ```
4. Put the URL in `README.md` (replace `https://<your-service>.onrender.com` and `<github-user>`), commit and push.

## 4. Keep-alive (important on the free plan)

Render's free web services **sleep after 15 minutes without traffic**, and the next visitor waits about a minute for a cold start. The repo includes `.github/workflows/keepalive.yml`, which pings `/api/health` every 10 minutes. To turn it on:

GitHub repo → **Settings → Secrets and variables → Actions → Variables → New repository variable**: `APP_URL` = `https://<your-service>.onrender.com`.

- One service running all month is about 744 hours, inside Render's 750 free instance-hours per workspace. Don't run a second free service in the same workspace with keep-alive on.
- GitHub may delay scheduled runs by a few minutes, and **disables scheduled workflows after 60 days without repository activity**. Re-enable from the Actions tab if needed.
- Right before sharing the link, open it once yourself.

## 5. Costs and the OpenRouter account

- **Models:** primary `deepseek/deepseek-v4-flash-0731`, fallback `openai/gpt-5.6-luna`. The fallback is only used if the primary errors. Your account's guardrail allows both and blocks all free models. If the guardrail changes, the app shows a data-only answer ("AI model busy…") instead of failing.
- **Spend:** about **$0.001 per answer**, roughly $1 per 1,000 answers. Pre-warm costs about $0.002 per restart (measured $0.0018 for the 6 suggested questions). The whole build, test and eval session cost **$0.12**.
- **Cost guard:** the app only ever calls models listed in `ALLOWED_PAID_MODELS` (default: the two above). To change models, set both `OPENROUTER_MODELS` and `ALLOWED_PAID_MODELS` in Render's Environment tab.
- **Abuse protection:** 10 requests/min per IP, a 1,000-character message limit, and an answer cache. A per-key credit limit in OpenRouter (Settings → Keys) adds a hard cap.
- After the review, consider rotating the key in OpenRouter and updating it in Render.

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Answers start with "The AI model is busy right now…" | The LLM call failed. Check Render **Logs** for `agentic:` warnings. **401** → wrong key, so re-enter it (the app strips surrounding spaces and quotes). **404 "guardrail"** → the account's guardrail no longer allows the configured models. **402** → out of credits. |
| Service restarts / "Out of memory" | Make sure `USE_RERANKER=false` and `ONNX_THREADS=1` are set on the free plan. |
| First request very slow | Cold start after sleep (~30–60 s at 0.1 CPU). Enable the keep-alive (section 4). |
| Build fails downloading models | Hugging Face hiccup; re-run the deploy (**Manual Deploy → Clear build cache & deploy**). |
| `429 Too many requests` in the UI | The per-IP rate limit (10/min). Wait a minute, or raise `RATE_LIMIT` in Render's Environment. |
