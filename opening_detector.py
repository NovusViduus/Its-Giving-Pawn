"""
opening_detector.py - My bot's opening moveset detector

Basically looks to see if the opposing bot is using random moves or an actual
strategy and counters accordingly. It has a built in tolerance for hybrid strategies
where an opposing bot might throw in random moves to look like it isnt using a specific
opening strategy.

Author: Graeme Huntley
"""

import chess
from typing import Optional, List, Tuple, Dict


class CounterStrategy:
    """
    This class holds the information to counter an opponents opening moves.
    It uses a preference scoring system to find opposing moves that fit with a preexisiting
    opening strategy. 
    """
    
    def __init__(
        self,
        name: str,
        preferred_pieces: List[int],
        preferred_squares: List[int],
        bonus: int
    ):
        """
        This initializes my bot's counter-strategy.
        
        Arguments:
            name: The name of the counter-strategy
            preferred_pieces: A list of chess.PIECE_TYPE constants
            preferred_squares: A list of chess square constants
            bonus: The score bonus for moves matching preferences
        """
        self.name = name
        self.preferred_pieces = preferred_pieces
        self.preferred_squares = preferred_squares
        self.bonus = bonus
    
    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Finds the best move fitting a specific counter-strategy by scoring them based on:
        - if they move the strategy's prefered piece type
        - if the piece moves to the strategy's prefered square
        - combinations of the two
        
        it uses the current board set up as an argument and returns the best counter move or
        nothing if there isn't a good match
        """
        legal_moves = list(board.legal_moves)
        move_scores = []
        
        for move in legal_moves:
            score = 0
            piece = board.piece_at(move.from_square)
            
            if piece:
                if piece.piece_type in self.preferred_pieces:
                    score += self.bonus
                
                if move.to_square in self.preferred_squares:
                    score += self.bonus
            
            if score > 0:
                move_scores.append((score, move))
        
        if move_scores:
            move_scores.sort(key=lambda x: x[0], reverse=True)
            return move_scores[0][1]
        
        return None
    
    def __repr__(self) -> str:
        """String representation strictly for debugging :D"""
        return f"CounterStrategy('{self.name}', bonus={self.bonus})"


class OpeningDetector:
    """
    Detects enemy bot opening patterns and suggests countering moves. It only checks the 3rd-12th moves
    against each known pattern then calculates a matching score. It also allows for tolerating early moves
    in case the opponent tries to be sneaky and throw it off. Also checks its self for consistancy before locking
    in when about 60% of the recent checks match a particular pattern.

    For attributes it uses a combination of Dicts to hold known opening patterns and counter strategies, a
    list to hold the opponents moves, an int variable to track the opponets move count, a tuple to act like a 
    detection window, and a bool flag for if a pattern is locked in by the detector or not.
    """
    
    def __init__(self, detection_window: Tuple[int, int] = (3, 12)):
        self.detection_start, self.detection_end = detection_window     
        self.patterns = self._define_patterns()
        self.counters = self._define_counters()
        self.opponent_moves = []
        self.move_number = 0
        self.pattern_history = []
        self.current_pattern = None
        self.pattern_locked_in = False
        self.pattern_entry_move = None
        self.early_move_forgiveness = 2
    
    def _define_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Defines the most common opening move sets in chess as white/black pairs
        """
        return {
            'italian_game': [
                ('e2e4', 'e7e5'),
                ('g1f3', 'b8c6'),
                ('f1c4', 'f8c5')
            ],
            'ruy_lopez': [
                ('e2e4', 'e7e5'),
                ('g1f3', 'b8c6'),
                ('f1b5', 'a7a6')
            ],
            'queens_gambit': [
                ('d2d4', 'd7d5'),
                ('c2c4', 'e7e6'),
                ('b1c3', 'g8f6')
            ],
            'london_system': [
                ('d2d4', 'd7d5'),
                ('c1f4', 'g8f6'),
                ('e2e3', 'e7e6'),
                ('g1f3', 'c7c5')
            ],
            'kings_indian_attack': [
                ('g1f3', 'g8f6'),
                ('g2g3', 'g7g6'),
                ('f1g2', 'f8g7'),
                ('e1g1', 'e8g8')
            ],
            'sicilian_defense': [
                ('e2e4', 'c7c5'),
                ('g1f3', 'd7d6'),
                ('d2d4', 'c5d4')
            ],
            'french_defense': [
                ('e2e4', 'e7e6'),
                ('d2d4', 'd7d5'),
                ('b1c3', 'g8f6')
            ],
            'caro_kann': [
                ('e2e4', 'c7c6'),
                ('d2d4', 'd7d5'),
                ('b1c3', 'd5e4')
            ],
        }
    
    def _define_counters(self) -> Dict[str, CounterStrategy]:
        """
        Defines the counter-moves for each opening listed in the prior function
        """
        return {
            'italian_game': CounterStrategy(
                name='two_knights_defense',
                preferred_pieces=[chess.KNIGHT],
                preferred_squares=[chess.F6, chess.C6],
                bonus=200
            ),
            'ruy_lopez': CounterStrategy(
                name='marshall_attack',
                preferred_pieces=[chess.PAWN],
                preferred_squares=[chess.D5, chess.F5],
                bonus=150
            ),
            'queens_gambit': CounterStrategy(
                name='slav_defense',
                preferred_pieces=[chess.PAWN],
                preferred_squares=[chess.C6, chess.C4],
                bonus=150
            ),
            'london_system': CounterStrategy(
                name='kings_indian_setup',
                preferred_pieces=[chess.BISHOP, chess.PAWN],
                preferred_squares=[chess.G7, chess.G6],
                bonus=150
            ),
            'kings_indian_attack': CounterStrategy(
                name='french_structure',
                preferred_pieces=[chess.PAWN],
                preferred_squares=[chess.E6, chess.D5],
                bonus=150
            ),
            'sicilian_defense': CounterStrategy(
                name='open_sicilian',
                preferred_pieces=[chess.KNIGHT, chess.PAWN],
                preferred_squares=[chess.F3, chess.D4, chess.C3],
                bonus=150
            ),
            'french_defense': CounterStrategy(
                name='advance_variation',
                preferred_pieces=[chess.PAWN],
                preferred_squares=[chess.E5, chess.C3],
                bonus=150
            ),
            'caro_kann': CounterStrategy(
                name='advance_variation',
                preferred_pieces=[chess.PAWN, chess.KNIGHT],
                preferred_squares=[chess.E5, chess.F3],
                bonus=150
            ),
        }
    
    def update(self, move: chess.Move, is_opponent: bool) -> None:
        """
        Updates the bot's detector with a new move from the game.
        """
        if is_opponent:
            self.opponent_moves.append(move.uci())
            self.move_number = len(self.opponent_moves)
            
            if self.detection_start <= self.move_number <= self.detection_end:
                self._update_pattern_detection()
    
    def _update_pattern_detection(self) -> None:
        """
        Runs the pattern detecter and checks for consistency. This is called each time the opponent moves
        during the detection window.
        """
        detected_pattern, confidence = self._detect_current_pattern()
        
        self.pattern_history.append({
            'move': self.move_number,
            'pattern': detected_pattern,
            'confidence': confidence
        })
        
        if self._is_pattern_consistent(detected_pattern):
            if not self.pattern_locked_in:
                print(f"   📚 Pattern LOCKED: {detected_pattern} at move {self.move_number}")
                self.current_pattern = detected_pattern
                self.pattern_locked_in = True
                self.pattern_entry_move = self.move_number
    
    def _detect_current_pattern(self) -> Tuple[Optional[str], float]:
        """
        Detects which pattern best matches the moves made by the opponent
        """
        if self.move_number < self.detection_start:
            return None, 0.0
        
        best_pattern = None
        best_score = 0.0
        
        for pattern_name, move_sequences in self.patterns.items():
            score = self._flexible_pattern_match(move_sequences)
            if score > best_score:
                best_score = score
                best_pattern = pattern_name
        
        return best_pattern, best_score
    
    def _flexible_pattern_match(self, pattern_sequences: List[Tuple[str, str]]) -> float:
        """
        This is used by my bot to handle opposing bots that might throw in random moves in the opening stage of the game
        to disrupt pattern matching.
        """
        best_score = 0.0
        
        for start_offset in range(min(self.early_move_forgiveness + 1, len(self.opponent_moves))):
            score = self._match_from_offset(pattern_sequences, start_offset)
            best_score = max(best_score, score)
        
        return best_score
    
    def _match_from_offset(
        self,
        pattern_sequences: List[Tuple[str, str]],
        offset: int
    ) -> float:
        """
        Uses the offest as the number of moves to skip ie if the opponent does a few random moves, this ensures
        the dector will start detecting AFTER the random moves finish.
        """
        matches = 0
        total = 0
        opponent_idx = offset
        
        expected_moves = [black_move for _, black_move in pattern_sequences]
        
        for expected_move in expected_moves:
            if opponent_idx >= len(self.opponent_moves):
                break
            
            if self.opponent_moves[opponent_idx] == expected_move:
                matches += 1
            total += 1
            opponent_idx += 1
        
        return matches / total if total > 0 else 0.0
    
    def _is_pattern_consistent(self, detected_pattern: Optional[str]) -> bool:
        """
        Helps tackle false positives by making sure the pattern is really there before committing to a counter
        """
        if not detected_pattern:
            return False
        
        recent = self.pattern_history[-3:]
        if len(recent) < 2:
            return False
        
        pattern_count = sum(1 for h in recent if h['pattern'] == detected_pattern)
        
        confidences = [h['confidence'] for h in recent if h['pattern'] == detected_pattern]
        avg_confidence = sum(confidences) / max(len(confidences), 1)
        
        return pattern_count >= 2 and avg_confidence >= 0.65
    
    def should_use_counter(self) -> bool:
        """
        Checks if my bot should use a counter strat finally
        """
        return (self.pattern_locked_in and 
                self.move_number <= self.detection_end)
    
    def get_current_pattern(self) -> Optional[str]:
        """
        Grabs the currently locked pattern for my bot
        """
        return self.current_pattern if self.pattern_locked_in else None
    
    def get_counter_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Gets the specific counter move for current board positoning
        """
        if not self.pattern_locked_in:
            return None
        
        counter = self.counters.get(self.current_pattern)
        if not counter:
            return None
        
        return counter.get_move(board)
    
    def reset(self) -> None:
        """Reset detector for new games"""
        self.opponent_moves = []
        self.move_number = 0
        self.pattern_history = []
        self.current_pattern = None
        self.pattern_locked_in = False
        self.pattern_entry_move = None
    
    def get_stats(self) -> dict:
        """
        A stats function for debuggin
        """
        return {
            'opponent_moves': len(self.opponent_moves),
            'current_pattern': self.current_pattern,
            'pattern_locked': self.pattern_locked_in,
            'pattern_entry_move': self.pattern_entry_move,
            'detection_history': len(self.pattern_history),
        }
    
    def __repr__(self) -> str:
        """String representation for debuggin"""
        return (f"OpeningDetector(pattern={self.current_pattern}, "
                f"locked={self.pattern_locked_in}, moves={self.move_number})")