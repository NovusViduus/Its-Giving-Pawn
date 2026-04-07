#!/usr/bin/env python3
"""
its_giving_pawn_uci.py - UCI Protocol Wrapper please do not separate from its_giving_pawn.py

This file wraps my chess bot in the UCI (Universal Chess Interface) protocol
so it can actually play against other engines and work with chess GUIs like
Arena or Cutechess. Without this, my bot is just code that can't talk to
anything else.

The tricky part here is that UCI communication has to be SUPER clean just
specific commands and responses but my bot loves to print stats and debug
info everywhere. So this file suppresses all my bot's chattiness while letting
UCI messages through properly.

Properly suppresses bot output but keeps UCI messages clean!

Author: Graeme Huntley
"""

import sys
import os
import chess
from its_giving_pawn import ChessBot


class SilentBot:
    """Context manager that suppresses my ChessBot's print statements without
    breaking UCI communication.
    
    This is necessary because UCI protocol is SUPER strict about what gets
    printed to stdout it only wants specific UCI commands and responses.
    My bot prints a ton of useful debug info (search stats, time management,
    etc.) but that would totally break UCI communication if it leaked through.
    
    So this redirects stdout to /dev/null while my bot is thinking, then
    restores it so UCI messages can go through. It's like putting my bot
    in a soundproof room while it thinks, then letting it whisper the move
    when it's done."""
    def __init__(self):
        self.devnull = None
        self.original_stdout = None
        
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.devnull = open(os.devnull, 'w')
        sys.stdout = self.devnull
        return self
    
    def __exit__(self, *args):
        if self.devnull:
            sys.stdout = self.original_stdout
            self.devnull.close()


class UCIEngine:
    """UCI protocol wrapper that makes my bot speak the universal chess language.
    
    This class handles all the UCI commands like 'uci', 'isready', 'position',
    'go', etc. and translates them into actions my bot can understand. It's
    basically the interpreter between chess GUIs and my actual bot code.
    
    Without this, my bot would just be sitting there unable to communicate
    with anything except manual Python function calls."""
    
    def __init__(self):
        self.bot = None
        self.board = chess.Board()
    
    def uci(self):
        """Responds to the 'uci' command with engine identification.
        
        This is the first command a GUI sends to figure out what engine it's
        talking to. We respond with our name and author, then 'uciok' to signal
        that we're ready to communicate via UCI protocol."""
        print("id name It's Giving Pawn")
        print("id author CS5100 Student")
        print("uciok")
        sys.stdout.flush()
    
    def isready(self):
        """Responds to 'isready' command and initializes the bot if needed.
        
        This is where the bot actually gets created (lazy initialization). I do
        this here instead of in __init__ because it saves startup time - the bot
        takes a second to load the opening book, evaluator, etc., and we don't
        want to do that until we actually need it.
        
        All the bot's initialization output gets suppressed so it doesn't mess
        up UCI communication. Once ready, we respond with 'readyok'"""
        if self.bot is None:
            with SilentBot():
                self.bot = ChessBot(
                    book_file="performance.bin",
                    time_budget=600.0,
                    use_trained_weights=True
                )
        print("readyok")
        sys.stdout.flush()
    
    def ucinewgame(self):
        """ Starts a new game by resetting the board and bot state.
        
        This clears out all the transposition tables, move history, time tracking,
        and other stuff from the previous game so the bot starts fresh. It's like
        hitting the reset button between games in a tournament."""
        self.board = chess.Board()
        if self.bot:
            with SilentBot():
                self.bot.reset()
    
    def position(self, args):
        """ Sets up the board position from UCI position command.
        
        This parses position commands which can be either:
        - 'position startpos moves e2e4 e7e5 ...' (starting position + moves)
        - 'position fen <fen string> moves ...' (custom position + moves)
        
        It reconstructs the board state by either starting from the initial
        position or a FEN string, then applies all the moves in order. This
        is how the GUI tells my bot what position it needs to analyze."""
        if "startpos" in args:
            self.board = chess.Board()
            moves_idx = args.index("moves") if "moves" in args else None
        elif "fen" in args:
            fen_idx = args.index("fen")
            moves_idx = args.index("moves") if "moves" in args else None
            
            if moves_idx:
                fen_parts = args[fen_idx + 1:moves_idx]
            else:
                fen_parts = args[fen_idx + 1:]
            
            fen_string = " ".join(fen_parts)
            self.board = chess.Board(fen_string)
        else:
            moves_idx = None
        
        if moves_idx is not None and moves_idx + 1 < len(args):
            for move_str in args[moves_idx + 1:]:
                try:
                    move = chess.Move.from_uci(move_str)
                    self.board.push(move)
                except:
                    pass
    
    def go(self, args):
        """Searches for the best move with the given time controls.
        
        This is where the actual thinking happens. It parses time control
        parameters like wtime (white time), btime (black time), winc (white
        increment), binc (black increment), or movetime (fixed time per move).
        
        Then it calculates a reasonable time limit for this specific move,
        tells my bot to search (with all output suppressed), and returns the
        best move in UCI format.
        
        The time calculation is important because going over time = instant loss,
        but being too conservative means weaker moves. I use a simple heuristic:
        divide remaining time by 30 moves, add the increment, and leave a 0.5s
        buffer for safety."""
        wtime = None
        btime = None
        winc = 0
        binc = 0
        movetime = None
        
        i = 0
        while i < len(args):
            if args[i] == "wtime" and i + 1 < len(args):
                wtime = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "btime" and i + 1 < len(args):
                btime = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "winc" and i + 1 < len(args):
                winc = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "binc" and i + 1 < len(args):
                binc = int(args[i + 1]) / 1000.0
                i += 2
            elif args[i] == "movetime" and i + 1 < len(args):
                movetime = int(args[i + 1]) / 1000.0
                i += 2
            else:
                i += 1
        
        if movetime:
            time_limit = movetime
        else:
            our_time = wtime if self.board.turn == chess.WHITE else btime
            our_inc = winc if self.board.turn == chess.WHITE else binc
            
            if our_time is None:
                time_limit = 5.0
            else:
                time_limit = (our_time / 30.0) + our_inc
                time_limit = max(0.1, min(time_limit, our_time - 0.5))
        
        with SilentBot():
            best_move = self.bot.get_move(self.board, time_limit=time_limit)
    
        if best_move is None:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
        
        if best_move:
            print(f"bestmove {best_move.uci()}")
            sys.stdout.flush()
        else:
            print("bestmove 0000")
            sys.stdout.flush()
    
    def quit(self):
        """Exits the engine cleanly.
        
        This is called when the GUI sends the 'quit' command. Just exits the
        program gracefully. Nothing fancy needed here."""
        sys.exit(0)
    
    def run(self):
        """Main UCI loop that listens for commands and responds.
        
        This is the heart of the UCI protocol implementation. It continuously
        reads commands from stdin, parses them, routes them to the appropriate
        handler functions, and keeps going until it receives 'quit' or EOF.
        
        I have exception handling here because UCI communication can be fragile
        and I don't want the engine to crash if something weird happens. Better
        to silently ignore a malformed command than to crash mid-tournament."""
        while True:
            try:
                line = input().strip()
                
                if not line:
                    continue
                
                parts = line.split()
                if not parts:
                    continue
                    
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []
                
                if command == "uci":
                    self.uci()
                elif command == "isready":
                    self.isready()
                elif command == "ucinewgame":
                    self.ucinewgame()
                elif command == "position":
                    self.position(args)
                elif command == "go":
                    self.go(args)
                elif command == "quit":
                    self.quit()
                    
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception:
                pass


if __name__ == "__main__":
    engine = UCIEngine()
    engine.run()