"""
FastAPI wrapper for It's Giving Pawn chess engine.
Exposes a /move endpoint that accepts a FEN string and returns the best move.

To run locally: uvicorn api:app --reload
To deploy: push to Railway with the included Procfile and requirements.txt
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess
import sys
import os
import time

# Add the current directory to path so we can import the engine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from its_giving_pawn import ItsGivingPawn

app = FastAPI(
    title="It's Giving Pawn API",
    description="Play against Graeme Huntley's chess engine",
    version="1.0.0"
)

# Allow CORS from your portfolio site (update with your actual domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this to your portfolio URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine once at startup
engine = None

@app.on_event("startup")
def startup():
    global engine
    engine = ItsGivingPawn()
    print("Engine initialized and ready.")


class MoveRequest(BaseModel):
    fen: str
    depth: int = 4  # Default search depth, keep low for fast responses
    time_limit: float = 3.0  # Max seconds to think


class MoveResponse(BaseModel):
    move: str  # UCI notation e.g. "e2e4"
    fen_after: str  # Board state after the move
    evaluation: int | None = None  # Centipawn evaluation if available
    thinking_time: float  # How long the engine took


@app.get("/")
def root():
    return {
        "name": "It's Giving Pawn",
        "author": "Graeme Huntley",
        "description": "UCI-compliant chess engine with 12-component evaluation",
        "endpoints": {
            "/move": "POST - Get best move for a position",
            "/health": "GET - Check if engine is running",
        }
    }


@app.get("/health")
def health():
    return {"status": "alive", "engine_loaded": engine is not None}


@app.post("/move", response_model=MoveResponse)
def get_move(request: MoveRequest):
    """
    Given a FEN string, return the engine's best move.
    """
    global engine

    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Validate the FEN
    try:
        board = chess.Board(request.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    if board.is_game_over():
        raise HTTPException(status_code=400, detail="Game is already over")

    # Get the engine's move
    start_time = time.time()
    try:
        # The engine should have a method to get the best move
        # Adjust this based on your actual engine interface
        best_move = engine.get_best_move(
            board,
            depth=request.depth,
            time_limit=request.time_limit
        )
    except Exception as e:
        # Fallback: if the engine interface differs, try alternative methods
        try:
            # Try UCI-style interface
            engine.set_position(board)
            best_move = engine.search(depth=request.depth)
        except Exception as e2:
            raise HTTPException(
                status_code=500,
                detail=f"Engine error: {str(e)}. Fallback error: {str(e2)}"
            )

    elapsed = time.time() - start_time

    # Apply the move to get the resulting FEN
    move_uci = str(best_move)
    board.push(best_move)

    return MoveResponse(
        move=move_uci,
        fen_after=board.fen(),
        thinking_time=round(elapsed, 3),
    )


@app.post("/new_game")
def new_game():
    """Reset the engine state for a new game."""
    global engine
    engine = ItsGivingPawn()
    return {"status": "new game ready", "fen": chess.STARTING_FEN}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
