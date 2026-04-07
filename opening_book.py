"""
opening_book.py - My bot's opening book handler

Manages opening book lookups using Polyglot opening book format (.bin files).
Basically this gives my bot instant grandmaster moves during the opening phase
so it doesn't waste time thinking about well-known positions. This saves a ton
of computation time and makes the bot play way better in the opening.

Author: Graeme Huntley
"""

import chess
import chess.polyglot
import os
from pathlib import Path
from typing import Optional, List


class OpeningBook:
    """
    This class handles opening book lookups using the Polyglot format.
    
    Basically the Polyglot format stores moves with weights based on how often
    they were played in high-level games. This is super useful because:
      - I get instant move lookup with zero search time
      - Moves come from master games so they're actually good
      - There's variety since it uses weighted random selection
    
    It's kinda like having a cheat sheet for the opening phase that's based
    on thousands of grandmaster games.
    """
    
    def __init__(
        self, 
        book_file: Optional[str] = None,
        fallback_books: Optional[List[str]] = None
    ):
        """
        Initializes my opening book handler with support for fallback books
        in case the primary one isn't found. This way my bot doesn't break
        if someone's missing a specific book file.
        """
        if book_file:
            self.book_files = [book_file]
        else:
            self.book_files = []
        
        if fallback_books:
            self.book_files.extend(fallback_books)
        
        if not self.book_files:
            self.book_files = [
                "performance.bin",  
                "human.bin",        
                "computer.bin",     
                "book.bin",         
            ]
        
        self.available_books = self._find_available_books()
        self.active_book = self.available_books[0] if self.available_books else None
        self.hits = 0  
        self.misses = 0  
        
        self._print_status()
    
    def _find_available_books(self) -> List[str]:
        """
        This searches for available opening book files in a bunch of common locations.
        
        I check multiple directories because people might put the book file in different
        spots and I don't want my bot to fail just because of a path issue. It looks in:
          1. Same directory as this file
          2. Current working directory
          3. 'books/' subdirectory
          4. 'opening_books/' subdirectory
        """
        found_books = []
        script_dir = Path(__file__).parent.absolute()
        search_dirs = [
            script_dir,                    
            Path.cwd(),                    
            script_dir / "books",          
            script_dir / "opening_books", 
        ]
 
        for book_file in self.book_files:
            for search_dir in search_dirs:
                book_path = search_dir / book_file
                if book_path.exists() and book_path.is_file():
                    found_books.append(str(book_path))
                    break  
        
        return found_books
    
    def _print_status(self) -> None:
        """Prints a nice status message about whether the opening book loaded successfully.
        If it didn't find one, it tells you how to download it so you're not stuck wondering
        why the bot isn't using book moves."""
        if self.active_book:
            book_name = Path(self.active_book).name
            print(f"Opening book loaded: {book_name}")
        else:
            print("No opening book found")
            print("To download:")
            print("wget https://github.com/michaeldv/donna_opening_books/komodo.bin")
            print("Bot will use search instead of book")
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Gets a move from the opening book for the current position using weighted
        random selection. Basically it picks moves based on how often they were
        played in master games - popular moves are more likely to be chosen.
        
        Returns None if the position isn't in the book, which is totally normal
        once you get past the opening phase.
        """
        if not self.active_book:
            self.misses += 1
            return None
        
        try:
            with chess.polyglot.open_reader(self.active_book) as reader:
                entry = reader.weighted_choice(board)
                self.hits += 1
                return entry.move
                
        except (FileNotFoundError, OSError) as e:
            print(f"Error reading book: {e}")
            self.active_book = None  
            self.misses += 1
            return None
            
        except IndexError:
            self.misses += 1
            return None
    
    def get_all_moves(self, board: chess.Board) -> List[tuple]:
        """
        Gets ALL the book moves for a position along with their weights and win rates.

        This is super useful for analysis or if you want to choose moves based on
        specific criteria instead of just weighted random. Like maybe you want the
        most popular move, or the most aggressive one, or whatever.
        
        Returns a list of (move, weight, win_rate) tuples sorted by weight.
        """
        if not self.active_book:
            return []
        
        try:
            with chess.polyglot.open_reader(self.active_book) as reader:
                entries = list(reader.find_all(board))
                entries.sort(key=lambda e: e.weight, reverse=True)
                results = []
                for entry in entries:
                    total_games = entry.learn + entry.wins + entry.draws + entry.losses
                    if total_games > 0:
                        win_rate = (entry.wins + entry.learn * 0.5) / total_games
                    else:
                        win_rate = 0.5  
                    
                    results.append((entry.move, entry.weight, win_rate))
                
                return results
                
        except (FileNotFoundError, OSError, IndexError):
            return []
    
    def get_best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Gets the most popular move from the book - basically the one with the
        highest weight. Unlike get_move() which uses weighted random selection,
        this always returns the single most frequently played move.
        
        Useful when you want consistency instead of variety.
        """
        moves = self.get_all_moves(board)
        if moves:
            return moves[0][0]
        return None
    
    def is_in_book(self, board: chess.Board) -> bool:
        """
        Simple check to see if the current position has any moves available
        in the opening book. Returns True if we've got book moves, False if not.
        """
        if not self.active_book:
            return False
        
        try:
            with chess.polyglot.open_reader(self.active_book) as reader:
                entries = list(reader.find_all(board))
                return len(entries) > 0
        except (FileNotFoundError, OSError, IndexError):
            return False
    
    def get_book_depth(self, board: chess.Board) -> int:
        """
         Figures out how many more moves the book covers from the current position
        by following the main line ie most popular moves 
        
        This is helpful for knowing when you're about to exit book and need to
        start actually thinking. I cap it at 30 moves because that's reasonable
        and prevents infinite loops if something weird happens
        """
        if not self.active_book or not self.is_in_book(board):
            return 0
        
        test_board = board.copy()
        depth = 0
        max_depth = 30 
        
        while depth < max_depth:
            move = self.get_best_move(test_board)
            if not move:
                break
            
            test_board.push(move)
            depth += 1
        
        return depth
    
    def get_hit_rate(self) -> float:
        """
        Calculates what percentage of my book lookups were successful. Basically
        just divides hits by total attempts to see how often positions were
        actually in the book.
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def get_stats(self) -> dict:
        """
        Returns all the opening book stats in a nice dictionary format for
        debugging and analysis. Shows things like hit rate, total lookups,
        which book is active etc
        """
        return {
            'active_book': Path(self.active_book).name if self.active_book else None,
            'book_available': self.active_book is not None,
            'hits': self.hits,
            'misses': self.misses,
            'total_lookups': self.hits + self.misses,
            'hit_rate': self.get_hit_rate(),
        }
    
    def print_stats(self) -> None:
        """Nice print format for opening book statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("OPENING BOOK STATISTICS")
        print("="*50)
        print(f"Active book:     {stats['active_book'] or 'None'}")
        print(f"Book available:  {'Yes' if stats['book_available'] else 'No'}")
        print(f"Hits:            {stats['hits']}")
        print(f"Misses:          {stats['misses']}")
        print(f"Total lookups:   {stats['total_lookups']}")
        print(f"Hit rate:        {stats['hit_rate']*100:.1f}%")
        print("="*50 + "\n")
    
    def __repr__(self) -> str:
        """String representation for debuggin"""
        book_name = Path(self.active_book).name if self.active_book else "None"
        return f"OpeningBook(active='{book_name}', hits={self.hits}, misses={self.misses})"


def download_book(book_name: str = "performance.bin") -> bool:
    """
    Helper function to download an opening book specifically komodo. I made this because manually downloading books is annoying
    and this way people can just run the script and get set up automatically.
    
    Returns True if it worked, False if something went wrong.

    """
    import urllib.request
    
    base_url = "https://github.com/michaeldv/donna_opening_books/komodo.bin"
    url = base_url + book_name
    
    try:
        print(f"Downloading {book_name}...")
        urllib.request.urlretrieve(url, book_name)
        print(f"Downloaded successfully to {book_name}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"\nManual download:")
        print(f"  wget {url}")
        return False
