# It's Giving Pawn — API Deployment Guide

## How to deploy your chess engine as an API on Railway

### Step 1: Prep your repo
Copy these files into your Its-Giving-Pawn repo root:
- `api.py` (the FastAPI wrapper)
- `requirements.txt` (dependencies)
- `Procfile` (tells Railway how to run the app)

Your repo should look like:
```
Its-Giving-Pawn/
├── api.py                  ← NEW
├── requirements.txt        ← NEW (or update existing)
├── Procfile                ← NEW
├── its_giving_pawn.py
├── its_giving_pawn_uci.py
├── search_engine.py
├── evaluator.py
├── generator.py
├── opening_book.py
├── opening_detector.py
├── time_manager.py
├── performance.bin
└── ... other files
```

### Step 2: Adjust the API to match your engine's interface
The `api.py` file assumes your engine has a `get_best_move(board, depth, time_limit)` method.
You may need to adjust the import and method call based on how your engine actually works.

Open `its_giving_pawn.py` and check:
- What class name you use (I assumed `ItsGivingPawn`)
- What method returns the best move
- What parameters it takes

Update the import and the `get_move()` function in `api.py` accordingly.

### Step 3: Test locally
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Then visit http://localhost:8000 — you should see the API info.

Test with curl:
```bash
curl -X POST http://localhost:8000/move \
  -H "Content-Type: application/json" \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "depth": 4}'
```

### Step 4: Deploy to Railway
1. Go to https://railway.app and sign up with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your Its-Giving-Pawn repo
4. Railway will auto-detect the Procfile and deploy
5. Once deployed, you'll get a URL like `https://its-giving-pawn-production.up.railway.app`
6. Test it: visit `https://your-url.up.railway.app/health`

### Step 5: Connect to your portfolio
Update the chess component in your portfolio to call:
```
POST https://your-railway-url.up.railway.app/move
Body: { "fen": "...", "depth": 4 }
```

### Notes
- Railway free tier gives you 500 hours/month — plenty for a portfolio
- Keep depth at 3-4 for fast responses (under 3 seconds)
- The engine loads once at startup, so first request after a cold start may be slow
- Add your actual portfolio domain to CORS origins in api.py for production
