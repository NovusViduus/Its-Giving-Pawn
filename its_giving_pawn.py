"""
its_giving_pawn.py - Main Chess Bot Orchestrator

This is the main file that brings all my chess bot components together into
one cohesive system. It handles the overall game flow, decides when to use
the opening book vs search, manages time budgets, and makes sure my bot never
goes over the 60 second time limit.

Author: Graeme Huntley
"""

import chess
import time
from pathlib import Path
from typing import Optional

from opening_book import OpeningBook
from opening_detector import OpeningDetector
from search_engine import SearchEngine
from evaluator import Evaluator
from time_manager import TimeManager


class ChessBot:
    """ This is the main chess bot class that orchestrates everything.
    
    Basically it coordinates between the opening book, pattern detector, search engine,
    evaluator, and time manager to make intelligent move decisions. It also has
    learned weights support so if I've trained the bot against Stockfish, it'll
    automatically load those improved weights instead of using defaults.

    -UPDATE I ditched the training portion due to time constraints
    
    The bot follows a clear strategy:
    - Opening phase (moves 1-15): Use opening book for instant grandmaster moves
    - Mid/Endgame: Use deep search with the evaluation function
    - Emergency mode: Fast moves when running low on time"""
    
    def __init__(
        self, 
        detection_window=(3, 12),
        time_budget=60.0,
        book_file="performance.bin",
        use_trained_weights=True
    ):
        """
        Initializes my chess bot with all its components.
        
        This sets up the opening book, pattern detector, evaluator, search engine,
        and time manager. If trained weights are available from machine learning
        training sessions, it automatically loads them instead of using the default
        hand-tuned weights.
        
        I spent way too long debugging time management issues so now this has
        multiple safety checks to ensure we never go over time during tournament play.
        """
        self.name = "it's_giving_pawn"
        self.move_count = 0
        
        print(f"Initializing {self.name}...")
        
        self.opening_book = OpeningBook(book_file=book_file)
        print("Opening book loaded")
        
        self.opening_detector = OpeningDetector(detection_window=detection_window)
        print("Pattern detector online, all systems optimal")
        
        if use_trained_weights:
            weights_file = "training_data/weights.json"
            if Path(weights_file).exists():
                self.evaluator = Evaluator(weights_file=weights_file)
                print("Loaded trained weights!")
            else:
                self.evaluator = Evaluator()
                print("No trained weights found (using defaults)")
        else:
            self.evaluator = Evaluator()
            print("Evaluator initialized (default weights)")
        
        self.search_engine = SearchEngine(evaluator=self.evaluator)
        print("Search engine ready")
        
        self.time_manager = TimeManager(total_budget=time_budget)
        print("Time manager configured")
        
        print(f"{self.name} ready for battle!\n")
    
    def get_move(self, board: chess.Board, time_limit: float = 5.0) -> chess.Move:
        """Main entry point for getting the bot's next move with hard time enforcement.
        
        This is the most critical function in the whole bot because it has to:
        - Check if we're running out of time and go into emergency mode if needed
        - Update the pattern detector with opponent moves
        - Try the opening book first moves 1-15 to save computation time
        - Fall back to deep search when book doesn't have the position
        - Track time usage so we never exceed the 60 second total budget
        
        I have multiple safety layers here because going over time = instant loss
        in the tournament, which would be super embarrassing after all this work.
        The emergency modes kick in at 8s and 3s remaining to ensure we always
        have time to make a move, even if it's not the best one."""
        
        time_remaining = self.time_manager.get_time_remaining()
        
        if time_remaining < 3.0:
            print("🚨 EMERGENCY MODE - < 3s remaining!")
            legal_moves = list(board.legal_moves)
            for move in legal_moves:
                if board.is_capture(move) or board.gives_check(move):
                    self.move_count += 1
                    self.time_manager.record_time(0.1)
                    return move
            self.move_count += 1
            self.time_manager.record_time(0.1)
            return legal_moves[0] if legal_moves else None
        
        if time_remaining < 8.0:
            print("⚡ PANIC MODE - < 8s remaining!")
            time_limit = 0.3  
        
        time_limit = min(time_limit, time_remaining - 2.0)
        if time_limit < 0.1:
            time_limit = 0.1
        
        move_start_time = time.time()
        self.move_count += 1
        
        print(f"\n{'='*60}")
        print(f"MOVE {self.move_count} - {'WHITE' if board.turn else 'BLACK'}")
        print(f"{'='*60}")
        
        self._update_detector(board)
        
        if self.move_count <= 15:
            book_move = self._try_opening_book(board)
            if book_move:
                self._finalize_move(move_start_time)
                return book_move
        
        best_move = self._search_for_move(board, time_limit)
        self._finalize_move(move_start_time)
        return best_move
    
    def _update_detector(self, board: chess.Board) -> None:
        """Updates the opening pattern detector with the most recent move.
        This feeds opponent moves into the detector so it can figure out if they're
        following a known opening pattern like the Italian Game or Ruy Lopez. Once
        it detects a pattern, the bot can use counter-strategies."""
        if len(board.move_stack) > 0:
            last_move = board.peek()
            is_opponent = True
            self.opening_detector.update(last_move, is_opponent)
    
    def _try_opening_book(self, board: chess.Board) -> Optional[chess.Move]:
        """Attempts to get a move from the opening book instead of searching.
        
        This is super helpful because opening book moves are instant (no computation)
        and they come from grandmaster games so they're actually good. It saves a ton
        of time during the opening phase which I can then use for deeper searches in
        the middlegame and endgame where positions get more complex.
        
        Returns the book move if found, None if the position isn't in the book."""
        book_move = self.opening_book.get_move(board)
        
        if book_move:
            pattern = self.opening_detector.get_current_pattern()
            
            if pattern:
                print(f"📖 Book move (detected opponent playing {pattern})")
            else:
                print(f"📖 Book move")
            
            print(f"   Move: {book_move}")
            return book_move
        
        return None
    
    def _search_for_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """ Runs the deep search engine to find the best move for the current position.
        
        This calculates a dynamic time budget based on game phase and time remaining,
        then runs iterative deepening search with all the fancy stuff like alpha-beta
        pruning, transposition tables, null move pruning, late move reductions, etc.
        
        Has a fallback to return the first legal move if search completely fails,
        though that should basically never happen unless something is seriously broken."""
        time_budget = self.time_manager.calculate_budget(self.move_count)
        time_budget = min(time_budget, time_limit)
        
        print(f"   Deep search")
        print(f"   Time budget: {time_budget:.2f}s")
        print(f"   Total used: {self.time_manager.time_used:.2f}s / {self.time_manager.total_budget:.2f}s")
        best_move = self.search_engine.iterative_deepening_search(board, time_budget)
    
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                raise ValueError("No legal moves available!")
            best_move = legal_moves[0]
            print(f"   Search failed, using fallback move: {best_move}")
        
        return best_move
    
    def _finalize_move(self, start_time: float) -> None:
        """Records how much time was used for the move and prints stats.
        
        This is critical for time management because it updates the time manager
        with actual elapsed time, which affects how much time future moves get.
        I use this data to make sure my bot doesn't run out of time before the
        game ends."""
        elapsed = time.time() - start_time
        self.time_manager.record_time(elapsed)
        
        print(f"     Time: {elapsed:.2f}s")
        print(f"     Total: {self.time_manager.time_used:.2f}s / {self.time_manager.total_budget:.2f}s")
        print(f"{'='*60}\n")
    
    def reset(self) -> None:
        """Resets my bot state for a new game.
        
        This clears out the move count, pattern detector history, transposition
        tables, evaluation cache, and time manager so the bot starts fresh for
        the next game. Basically like hitting the reset button on everything."""
        self.move_count = 0
        self.opening_detector.reset()
        self.search_engine.clear_tables()
        self.evaluator.clear_cache()
        self.time_manager.reset()
        print(f"{self.name} reset for new game")
    
    def get_stats(self) -> dict:
        """Returns all my bot's statistics in a dictionary format.
        
        This is useful for post-game analysis to see things like total time used,
        average time per move, and how much time is left. I use this data to
        tune the time management parameters."""
        return {
            'moves_played': self.move_count,
            'time_used': self.time_manager.time_used,
            'time_remaining': self.time_manager.total_budget - self.time_manager.time_used,
            'avg_time_per_move': self.time_manager.time_used / max(self.move_count, 1),
        }
    
    def print_stats(self) -> None:
        """Prints nicely formatted game statistics.
        
        I call this after games to see how the bot performed in terms of time
        management. It's helpful for debugging and for making sure my time
        allocation strategy is working as intended across different game lengths."""
        stats = self.get_stats()
        print("\n" + "="*60)
        print(f"{self.name.upper()} - GAME STATISTICS")
        print("="*60)
        print(f"Moves played:          {stats['moves_played']}")
        print(f"Time used:             {stats['time_used']:.2f}s")
        print(f"Time remaining:        {stats['time_remaining']:.2f}s")
        print(f"Avg time per move:     {stats['avg_time_per_move']:.2f}s")
        print("="*60 + "\n")


def main():
    """Demo function that shows off the bot playing a few moves.
    
    This is mainly for testing to make sure all the components are working
    together properly. It plays 5 moves and prints stats so you can see
    the bot in action without running a full game."""
    print("="*60)
    print("IT'S GIVING PAWN - CHESS BOT")
    print("="*60)
    print()
    
    bot = ChessBot(
        detection_window=(3, 12),
        time_budget=60.0,
        book_file="performance.bin"
    )
    
    board = chess.Board()
    
    print("Playing 5 moves as demo:")
    print(board)
    print()
    
    for i in range(5):
        move = bot.get_move(board, time_limit=3.0)
        print(f"Bot plays: {move}")
        board.push(move)
        print(board)
        print()
        
        if board.is_game_over():
            print("Game over!")
            break
    
    bot.print_stats()
    
    print("Demo complete! ✨")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "draw":
        import subprocess
        subprocess.run(["python3", "visualize_chess_tree.py", "--opening", "vienna_frankenstein"])
    else:
        main()