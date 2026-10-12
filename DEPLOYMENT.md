# Public deployment

The frontend works **standalone**: it parses production, transportation, and
game-theory problems and solves them in the browser (`frontend/src/lib/`,
verified against the HiGHS backend). Deploying only the frontend on Vercel is
enough for a fully working public site — no server, no API keys, no card.

The FastAPI backend (`backend/`) is **optional**. It adds GPT-6 Astra
understanding of free-form descriptions. The site tries `VITE_API_URL` first
and silently falls back to the built-in browser solver.

## 1. Push the project

Commit and push the application to GitHub. Do not commit `backend/.env` or any API keys.

## 2. Deploy the API on Render

1. Create a new Render Blueprint from this repository.
2. Render will read `render.yaml` and create `optiforge-api`.
3. Add these secret environment variables in Render:
   - `EXPLABS_API_KEY`
   - `SIMPLE_AI_API_KEY`
4. Deploy and confirm the health endpoint at `https://<render-api-host>/`.

## 3. Deploy the frontend on Vercel

1. Import the repository into Vercel.
2. Keep the repository root as the project root. `vercel.json` builds `frontend`.
3. Add this environment variable in Vercel for Production:
   - `VITE_API_URL=https://<render-api-host>`
4. Deploy and verify that the frontend can call the API.

### No-card backend: Hugging Face Space

1. Create a new Docker Space at https://huggingface.co/new-space.
   - Name it e.g. `optiforge-api`, pick **Docker** as the SDK, leave visibility Public (free).
2. In the Space, create/replace `README.md` with exactly this (it tells HF to use Docker on port 7860):

   ```markdown
   ---
   title: OptiForge API
   emoji: ⚙️
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   pinned: false
   ---

    # OptiSolve API
    FastAPI backend for optisolve.ai (production / transportation / game-theory solver).
    ```

3. Upload the repository's root `Dockerfile`, `.dockerignore`, and the whole `backend` folder to the Space
   (via the web UI: Add file > Upload files, or `git clone` the Space and copy them in).
4. Add this Space secret under Settings > Variables and secrets (Repository secrets):
    - `FRONTEND_ORIGINS=https://optisolve.ai,https://www.optisolve.ai,https://optiforge-ai.vercel.app`
    - Skip `EXPLABS_API_KEY` / `SIMPLE_AI_API_KEY` for now — production examples solve without keys.
5. Wait for the Space to build, then confirm `https://<space-owner>-<space-name>.hf.space/` returns `{"message":"OptiForge backend is working"}`.
6. Set Vercel's `VITE_API_URL` to that Hugging Face URL instead of a Render URL.

## 4. Connect `optisolve.ai` (custom domain)

`optisolve.ai` must be registered first (buy it at Porkbun, Cloudflare Registrar, or Namecheap —
`.ai` domains cost roughly $70–140/yr). Then:

1. In Vercel, open the frontend project → Settings → Domains → add `optisolve.ai` and `www.optisolve.ai`.
2. Vercel will show the required DNS records. At your registrar add:
    - `A` record for `@` → `76.76.21.21`
    - `CNAME` record for `www` → `cname.vercel-dns.com`
3. Wait for DNS propagation, then make `optisolve.ai` the primary domain and enable
   "Redirect `optiforge-ai.vercel.app` to primary domain" (a code-level redirect is also
   already configured in `vercel.json`, so the old address forwards automatically).

The Render API can remain on its `onrender.com` hostname. The browser only needs the Vercel domain, and the API CORS configuration in `render.yaml` allows both apex and `www` domains.

## 5. Production checks

- Open `https://optisolve.ai`.
- Submit a simple game or production prompt and verify a result.
- Check the backend host logs for API errors.
- Confirm both provider keys are configured as hosting secrets and not in Git.
