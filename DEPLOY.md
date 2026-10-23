# Deploy for free

Your stack (FastAPI + Postgres 17 + pgvector + ~982 MB of indexed data + a
~176 MB model cache) rules out the usual free PaaS combos: Render's free
Postgres expired after 30 days, Neon/Supabase free tiers cap storage below
your DB size, and Hugging Face free Spaces sleep + wipe disk. The two
options below actually cost $0 and fit your app as-is.

## Option A — share from your PC (5 minutes, temporary link)

Best if you just want to show someone the site. Your PC stays the server.

```powershell
# 1. Make sure the app is running
docker compose up -d
# 2. In another terminal, expose it through a free Cloudflare Tunnel
#    (first time only: winget install --id Cloudflare.cloudflared)
cloudflared tunnel --url http://localhost:8000
```

Cloudflare prints a public `https://<random>.trycloudflare.com` URL. Send
that link. Stop it with Ctrl+C. Notes:

- Your PC + Docker must stay on while sharing.
- The free `trycloudflare.com` URL changes every restart. For a stable name,
  `cloudflared tunnel --hostname <your-domain>` needs a (free) Cloudflare
  account + a domain pointed at Cloudflare.

## Option B — Oracle Cloud Always Free VM (permanent, $0)

A free Ampere ARM VM (up to 4 OCPUs / 24 GB RAM, 200 GB disk on the free
tier) comfortably runs `docker compose up` with your DB. Stays up 24/7.

### 1. Push the repo to GitHub (without secrets)

```powershell
git init
git add Dockerfile docker-compose.yml .dockerignore .env.example requirements.txt app db frontend scripts README.md DEPLOY.md
git commit -m "deploy: ready for free VM"
git remote add origin <your-repo-url>
git push -u origin main
```

Never commit `.env` (it has your `GEMINI_API_KEY`). `.dockerignore` and
`.gitignore` already exclude it, `.sec_cache/`, logs, and `__pycache__/`.

### 2. Create the VM

1. Sign up at cloud.oracle.com (card verification, no charge for Always Free).
2. Compute → Instances → Create: image **Ubuntu 24.04**, shape **VM.Standard.A1.Flex**
   (ARM, e.g. 2 OCPUs / 12 GB RAM), boot volume 100 GB. Keep it **Always Free**
   eligible (the console labels it).
3. Networking: add ingress rules for TCP **80, 443** (and 8000 only while testing).
4. Save the private SSH key, note the public IP.

If you hit "out of host capacity", retry in a few hours or pick another
availability domain in your home region.

### 3. Install Docker on the VM

```bash
ssh -i <key> ubuntu@<PUBLIC_IP>
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker ubuntu && newgrp docker
```

### 4. Get the code + secrets onto the VM

```bash
git clone <your-repo-url> graphrag-sec && cd graphrag-sec
cp .env.example .env   # then edit: set GEMINI_API_KEY (and DATABASE_URL stays as-is)
nano .env
```

`docker-compose.yml` already interpolates `GEMINI_API_KEY`/`GEMINI_MODEL` from
`.env`, and the Dockerfile honors `$PORT` (defaults to 8000), so no code
changes are needed.

### 5. Move your data (don't re-ingest)

Re-ingesting 841 filings on a fresh VM takes hours and hammers SEC. Copy the
DB instead — one command on your PC:

```powershell
docker exec graphrag-sec-db-1 pg_dump -U graphrag -Fc graphrag > graphrag.dump
scp -i <key> graphrag.dump ubuntu@<PUBLIC_IP>:~/graphrag-sec/
```

On the VM:

```bash
cd ~/graphrag-sec && docker compose up -d db
# wait ~20s for a healthy DB, then restore (pgvector image already has the extension)
docker exec -i graphrag-sec-db-1 pg_restore -U graphrag -d graphrag --clean --if-exists < graphrag.dump
docker compose up -d --build
```

Check `http://<PUBLIC_IP>:8000/health` → `{"status":"ok"}`, then open
`http://<PUBLIC_IP>:8000/` in a browser.

### 6. (Recommended) stable HTTPS with one command

```bash
sudo apt-get install -y caddy
sudo caddy reverse-proxy --from <your-domain> --to localhost:8000
```

Caddy fetches a free Let's Encrypt cert automatically. Point your domain's A
record at the VM IP first.

## Free-tier caveats (read once)

- Oracle may reclaim **idle** Always Free VMs; log in occasionally or add a
  free uptime monitor (e.g. UptimeRobot) pinging `/health`.
- Back up the DB now and then: re-run the `pg_dump` step above and keep the
  `.dump` file somewhere safe (your 2 GB `.sec_cache/` lets you rebuild
  offline if you ever lose the DB).
- Your Gemini key was previously pasted in `.env.example`; that file now says
  `your-key-here`. If the old key ever touched a public repo, rotate it at
  https://aistudio.google.com/apikey and update `.env` (VM + PC).
- Don't run `uvicorn app.main:app` alongside `docker compose up` — both want
  port 8000.
