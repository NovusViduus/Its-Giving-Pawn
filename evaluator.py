"""
evaluator.py - Grandmaster evaluator

This is the brain of my chess bot that actually figures out if a position is good
or bad. It's way more sophisticated than just counting material, it understands
positional factors like king safety, pawn structure, piece coordination, and a
bunch of other stuff that separates strong engines from weak ones.

Strategic Philosophy:
Basically I built this to understand chess the way strong players do, not just
computers. It knows that evaluation is phase-dependent: king safety dominates the
middlegame, while king activity and pawn structure matter most in endgames. The
whole thing is designed to be tuned by machine learning against Stockfish. Althought I never
got around to using it to successfully beat stockfish at all even at leve zero tragically.

Features 12 tunable parameters for ML optimization:
1. Material + PST - Foundation with sophisticated piece-square tables
2. Piece Safety - Advanced SEE with hanging piece detection
3. King Safety - Multi-layered: pawn shield, attacker proximity, escape squares
4. Mobility - Quality-weighted mobility (not just move count)
5. Passed Pawns - Distance-based with blocker penalties
6. Bishop Pair - Openness-aware, phase-tapered, mobility-checked
7. Development - Castling, piece quality, tempo awareness
8. Center Control - Occupancy + influence with piece weighting
9. Rook Placement - Open files, 7th rank, battery formation
10. Pawn Structure - Islands, holes, backward pawns, phalanx
11. Knight Outposts - Protected advanced squares, enemy pawn analysis
12. Rook Connection - Horizontal/vertical coordination bonus

Author: Graeme Huntley
Version: 2.0 - With more broken hopes and dreams
"""

import chess
import chess.polyglot
import json
from pathlib import Path
from typing import Dict, List, Tuple


class Evaluator:
    """This is my grandmaster-level evaluator with phase-aware strategic understanding.
    
    Basically it combines 12 different strategic components that can all be tuned
    by machine learning. I spent forever researching proper chess evaluation
    techniques and implementing things like SEE (Static Exchange Evaluation),
    passed pawn evaluation, king safety metrics, and all the other positional
    factors that make a chess engine actually play well.
    
    The cool part is that everything is weighted and phase-dependent, so the
    bot knows what matters in the opening vs middlegame vs endgame. And if I
    train it against Stockfish, it automatically loads those optimized weights
    instead of my hand-tuned defaults."""
    
    def __init__(self, weights_file: str = None):
        print("Evaluator initialized")
        """" Initializes the evaluator with 12 tunable features and GM refinements.
        
        This sets up all the piece values, piece-square tables, evaluation weights,
        and the caching system. If there's a weights file from machine learning
        training, it automatically loads those improved weights instead of defaults.
        
        I use caching because evaluating the same position multiple times is a waste
        of computation, and in chess you see the same positions a LOT due to
        transpositions."""        
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,  
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        self.weights = {
            'material_pst_weight': 1.0,       
            'piece_safety_weight': 1.0,       
            'king_safety_weight': 1.0,        
            'mobility_weight': 1.0,           
            'passed_pawn_weight': 1.0,        
            'bishop_pair_weight': 1.0,        
            'development_weight': 1.0,        
            'center_control_weight': 1.0,     
            'rook_open_file_weight': 1.0,     
            'pawn_structure_weight': 1.0,     
            'knight_outpost_weight': 1.0,     
            'connected_rooks_weight': 1.0,    
        }
        
        if weights_file and Path(weights_file).exists():
            self._load_weights(weights_file)
            print(f"      📊 Loaded optimized weights from {weights_file}")
        
        self.eval_cache = {}
        self.max_cache_size = 50000
        self.cache_hits = 0
        self.cache_misses = 0
        self.positions_evaluated = 0
        self._init_pst()
        
        self.OPENING_THRESHOLD = 16  
        self.ENDGAME_THRESHOLD = 6   
    
    def _load_weights(self, filename: str):
        """Loads optimized weights from a JSON file created during ML training. No longer used sadly.
        
        This is how my bot gets smarter over time it plays against Stockfish,
        learns which weights work better, and saves them. Then next time it starts
        up, it uses those learned weights instead of my hand-tuned guesses."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'weights' in data:
            self.weights.update(data['weights'])
    
    def _init_pst(self):
        """
        Initializes sophisticated piece-square tables that encode positional preferences.
        
        These tables tell my bot where each piece type wants to be. Knights want to
        be centralized, bishops want long diagonals, rooks want open files and the
        7th rank, pawns want to advance, and kings want to be castled in the middlegame
        but centralized in the endgame.
        
        I spent a lot of time researching good PST values from strong engines and chess
        theory to make these tables actually useful instead of just random numbers.
        """
        
        self.knight_pst = [
            [-50, -40, -30, -30, -30, -30, -40, -50],
            [-40, -20,   0,   5,   5,   0, -20, -40],
            [-30,   0,  10,  15,  15,  10,   0, -30],
            [-30,   5,  15,  20,  20,  15,   5, -30],
            [-30,   5,  15,  20,  20,  15,   5, -30],
            [-30,   0,  10,  15,  15,  10,   0, -30],
            [-40, -20,   0,   0,   0,   0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50],
        ]
        
        self.bishop_pst = [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-10,  10,  10,  10,  10,  10,  10, -10],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-10,   5,   5,  10,  10,   5,   5, -10],
            [-10,   0,   5,  10,  10,   5,   0, -10],
            [-10,   0,   0,   0,   0,   0,   0, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20],
        ]
        
        self.rook_pst = [
            [ 0,  0,  0,  5,  5,  0,  0,  0],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [ 5, 10, 10, 10, 10, 10, 10,  5],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
        ]
        
        self.pawn_pst = [
            [  0,   0,   0,   0,   0,   0,   0,   0],
            [  5,   5,   5,   5,   5,   5,   5,   5],
            [  5,   5,  10,  15,  15,  10,   5,   5],
            [ 10,  10,  15,  25,  25,  15,  10,  10],
            [ 15,  15,  20,  30,  30,  20,  15,  15],
            [ 25,  25,  30,  40,  40,  30,  25,  25],
            [ 50,  50,  60,  70,  70,  60,  50,  50],
            [  0,   0,   0,   0,   0,   0,   0,   0],
        ]
        
        self.king_middlegame_pst = [
            [ 20,  30,  10,   0,   0,  10,  30,  20],
            [ 20,  20,   0,   0,   0,   0,  20,  20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
        ]
        
        self.king_endgame_pst = [
            [-50, -30, -30, -30, -30, -30, -30, -50],
            [-30, -10,   0,   0,   0,   0, -10, -30],
            [-30,   0,  20,  30,  30,  20,   0, -30],
            [-30,   0,  30,  40,  40,  30,   0, -30],
            [-30,   0,  30,  40,  40,  30,   0, -30],
            [-30,   0,  20,  30,  30,  20,   0, -30],
            [-30, -10,   0,   0,   0,   0, -10, -30],
            [-50, -30, -30, -30, -30, -30, -30, -50],
        ]
    
    def get_game_phase(self, board: chess.Board) -> float:
        """
        Calculates what phase of the game we're in: 1.0 = opening, 0.5 = middlegame, 0.0 = endgame.
        
        This is super important because different things matter at different stages of the game.
        In the opening you want development and king safety. In the middlegame you want activity
        and tactics. In the endgame you want king activity and passed pawns.
        
        I use a combination of move number and material on the board to figure this out, with
        smooth interpolation so there aren't sudden jumps that would make the evaluation unstable.
        """
        move_number = board.fullmove_number
        
        piece_count = 0
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            piece_count += len(board.pieces(piece_type, chess.WHITE))
            piece_count += len(board.pieces(piece_type, chess.BLACK))
        
        if piece_count <= self.ENDGAME_THRESHOLD or move_number > 40:
            return 0.0
        
        elif move_number <= 12 and piece_count >= self.OPENING_THRESHOLD:
            return 1.0
        
        else:
            material_phase = (piece_count - self.ENDGAME_THRESHOLD) / \
                           (self.OPENING_THRESHOLD - self.ENDGAME_THRESHOLD)
            material_phase = max(0.0, min(1.0, material_phase))
            
            move_phase = max(0.0, min(1.0, (40 - move_number) / 28.0))
            
            return material_phase * 0.8 + move_phase * 0.2
    
    def evaluate(self, board: chess.Board) -> int:
        """
        Main evaluation function that combines all 12 strategic components and returns
        a score in centipawns from the current player's perspective.
        
        This is the core function that gets called thousands of times per search to
        figure out if positions are good or bad. Positive = advantage for side to move,
        Negative = disadvantage.
        
        The magic here is that it checks the cache first (huge speed boost), then
        combines all the evaluation components with proper phase-aware weighting so
        the bot knows what actually matters in the current position. Terminal positions
        like checkmate get handled specially because those are forced wins/losses.
        """
        pos_hash = chess.polyglot.zobrist_hash(board)
        if pos_hash in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[pos_hash]
        
        self.cache_misses += 1
        self.positions_evaluated += 1
        
        if board.is_checkmate():
            return -100000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        phase = self.get_game_phase(board)
        is_endgame = phase < 0.3
        
        score = 0
        
        score += self._evaluate_material_and_pst(board, phase) * \
                 self.weights['material_pst_weight']
        
        score += self._evaluate_piece_safety(board) * \
                 self.weights['piece_safety_weight']
        
        if is_endgame:
            score += self._evaluate_king_activity(board)
        else:
            king_safety_score = self._evaluate_king_safety(board) * \
                              self.weights['king_safety_weight']
            score += int(king_safety_score * (phase ** 0.7))
        
        mobility_multiplier = 0.8 + (phase * 0.4) 
        score += self._evaluate_mobility(board) * \
                 self.weights['mobility_weight'] * mobility_multiplier
        
        passed_pawn_multiplier = 1.5 - (phase * 0.5)  
        score += self._evaluate_passed_pawns(board) * \
                 self.weights['passed_pawn_weight'] * passed_pawn_multiplier
        
        score += self._evaluate_bishop_pair(board) * \
                 self.weights['bishop_pair_weight']
        
        if phase > 0.4:  
            score += self._evaluate_development(board) * \
                     self.weights['development_weight']
        
        center_multiplier = 1.0 + (phase * 0.5) - ((1.0 - phase) * 0.3)
        score += self._evaluate_center_control(board) * \
                 self.weights['center_control_weight'] * center_multiplier
        
        rook_multiplier = 0.9 + ((1.0 - phase) * 0.3)
        score += self._evaluate_rook_open_files(board) * \
                 self.weights['rook_open_file_weight'] * rook_multiplier
        
        pawn_structure_multiplier = 0.85 + ((1.0 - phase) * 0.3)
        score += self._evaluate_pawn_structure(board) * \
                 self.weights['pawn_structure_weight'] * pawn_structure_multiplier
        
        knight_multiplier = 0.7 + (phase * 0.5)  
        score += self._evaluate_knight_outposts(board) * \
                 self.weights['knight_outpost_weight'] * knight_multiplier
        
        score += self._evaluate_connected_rooks(board) * \
                 self.weights['connected_rooks_weight']
        
        if len(self.eval_cache) < self.max_cache_size:
            self.eval_cache[pos_hash] = int(score)
        
        return int(score)
    
    def extract_features(self, board: chess.Board) -> List[float]:
        """
        Extracts ALL 12 features as normalized values for ML training.
        
        This is how the machine learning training system sees positions - as a list
        of 12 numbers representing different aspects of the position. The optimizer
        then learns which features are most important by playing tons of games against
        Stockfish and seeing which weights lead to better play.
        
        Returns a list of 12 normalized feature values that capture everything from
        material balance to pawn structure to king safety.

        update: sadly not really used :(
        """
        features = []
        
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = {
                    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.3,
                    chess.ROOK: 5, chess.QUEEN: 9
                }.get(piece.piece_type, 0)
                
                if piece.color == board.turn:
                    material += value
                else:
                    material -= value
        features.append(material)
        
        our_king = board.king(board.turn)
        castled = 0
        if our_king:
            if board.turn == chess.WHITE and our_king in [chess.G1, chess.C1]:
                castled = 1
            elif board.turn == chess.BLACK and our_king in [chess.G8, chess.C8]:
                castled = 1
        features.append(castled)
        
        our_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        enemy_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        mobility = (our_moves - enemy_moves) / 50.0
        features.append(mobility)
        
        passed = self._count_passed_pawns_simple(board)
        features.append(passed)
        
        development = self._count_development(board)
        features.append(development)
        
        our_bishops = len(board.pieces(chess.BISHOP, board.turn))
        enemy_bishops = len(board.pieces(chess.BISHOP, not board.turn))
        bishop_pair = 0
        if our_bishops >= 2:
            bishop_pair += 1
        if enemy_bishops >= 2:
            bishop_pair -= 1
        features.append(bishop_pair)
        
        center_control = self._count_center_attacks(board)
        features.append(center_control)
        
        rook_open = self._count_rooks_on_open_files(board)
        features.append(rook_open)
        
        pawn_structure = self._evaluate_pawn_structure_simple(board)
        features.append(pawn_structure)
        
        knight_outposts = self._count_knight_outposts(board)
        features.append(knight_outposts)
        
        connected_rooks = 1 if self._has_connected_rooks(board) else 0
        features.append(connected_rooks)
        
        threats = self._count_threats(board)
        features.append(threats)
        
        return features
    
    def _evaluate_material_and_pst(self, board: chess.Board, phase: float) -> int:
        """
       Evaluates material count plus piece-square table bonuses.
        
        This is the foundation of chess evaluation material matters! But it's not
        just about raw piece values, it's also about WHERE those pieces are positioned.
        A centralized knight is way more valuable than one stuck on the rim.
        
        PST values are phase-dependent for king positioning because kings want to hide
        in the middlegame but fight in the endgame. Returns centipawn evaluation from
        current player's perspective.
        """
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            
            value = self.piece_values[piece.piece_type]
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            eval_rank = rank if piece.color == chess.WHITE else (7 - rank)
            
            pst_bonus = 0
            if piece.piece_type == chess.KNIGHT:
                pst_bonus = self.knight_pst[eval_rank][file]
            elif piece.piece_type == chess.BISHOP:
                pst_bonus = self.bishop_pst[eval_rank][file]
            elif piece.piece_type == chess.ROOK:
                pst_bonus = self.rook_pst[eval_rank][file]
            elif piece.piece_type == chess.PAWN:
                pst_bonus = self.pawn_pst[eval_rank][file]
            elif piece.piece_type == chess.KING:
                mg_bonus = self.king_middlegame_pst[eval_rank][file]
                eg_bonus = self.king_endgame_pst[eval_rank][file]
                pst_bonus = int(mg_bonus * phase + eg_bonus * (1.0 - phase))
            
            total = value + pst_bonus
            
            if piece.color == board.turn:
                score += total
            else:
                score -= total
        
        return score
    
    def _evaluate_piece_safety(self, board: chess.Board) -> int:
        """
        Advanced piece safety using Static Exchange Evaluation (SEE).
        
        This detects hanging pieces and unfavorable trades, which is super important
        because leaving pieces undefended or in bad positions loses games quickly.
        SEE simulates the entire capture sequence to figure out if an exchange is
        winning or losing.
        
        Returns penalty for unsafe pieces from current player's perspective. I apply
        90% of the calculated penalty instead of 100% as a slight discount for the
        tactical complexity - sometimes "hanging" pieces have hidden purposes.
        """
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece or piece.color != board.turn or piece.piece_type == chess.KING:
                continue
            
            enemy_attackers = board.attackers(not board.turn, square)
            if not enemy_attackers:
                continue
            
            our_defenders = board.attackers(board.turn, square)
            
            see_score = self._see(board, square, piece.piece_type, 
                                 enemy_attackers, our_defenders)
            
            if see_score < 0:
                penalty = abs(see_score)
                score -= int(penalty * 0.9)
        
        return score
    
    def _see(self, board: chess.Board, square: int, piece_type: int,
             attackers, defenders) -> int:
        """
        Static Exchange Evaluation - calculates the material outcome if all captures
        are played out optimally on a given square.
        
        This is a critical function for tactical evaluation. It figures out if trading
        pieces on a square is good or bad by simulating the entire capture sequence
        with both sides capturing with their least valuable pieces first (standard
        capture strategy).
        
        Returns the final material balance: Negative = losing exchange, Positive = winning
        exchange, Zero = equal trade.
        """
        if not attackers:
            return 0
        
        attacker_values = []
        for sq in attackers:
            p = board.piece_at(sq)
            if p:
                attacker_values.append(self.piece_values[p.piece_type])
        
        defender_values = []
        for sq in defenders:
            p = board.piece_at(sq)
            if p:
                defender_values.append(self.piece_values[p.piece_type])
        
        attacker_values.sort()
        defender_values.sort()
        
        piece_value = self.piece_values[piece_type]
        gain = [piece_value]
        
        d = 0
        while d < len(attacker_values) and d < len(defender_values):
            if d % 2 == 0:  
                gain.append(attacker_values[d] - gain[-1])
            else:  
                gain.append(defender_values[d] - gain[-1])
            d += 1
        
        while len(gain) > 1:
            gain[-2] = -max(-gain[-1], gain[-2])
            gain.pop()
        
        return -gain[0]  
    
    def _evaluate_king_safety(self, board: chess.Board) -> int:
        """
        Comprehensive king safety evaluation for the middlegame.
        
        King safety is critical in the middlegame because one mistake can lead to a
        mating attack. This function looks at:
        - Castling status (castled = safe, uncastled = danger)
        - Pawn shield integrity (holes in the shield = bad)
        - Enemy piece proximity, especially queens
        - Attack intensity (how many enemy pieces are near the king)
        - Escape square availability (trapped king = disaster)
        
        I spent forever tuning these values because king safety is one of those things
        that separates good engines from mediocre ones. Returns safety score from
        current player's perspective.
        """
        score = 0
        
        our_king = board.king(board.turn)
        if not our_king:
            return 0
        
        king_file = chess.square_file(our_king)
        king_rank = chess.square_rank(our_king)
        
        has_castled = False
        if board.turn == chess.WHITE:
            has_castled = our_king in [chess.G1, chess.C1]
        else:
            has_castled = our_king in [chess.G8, chess.C8]
        
        if has_castled:
            score += 50
        else:
            piece_count = sum(len(board.pieces(pt, c)) 
                            for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                            for c in [chess.WHITE, chess.BLACK])
            if piece_count >= 12:  
                score -= 40
        
        pawn_direction = 1 if board.turn == chess.WHITE else -1
        shield_pawns = 0
        shield_holes = 0
        
        for file_offset in [-1, 0, 1]:
            for rank_offset in [1, 2]:
                target_file = king_file + file_offset
                target_rank = king_rank + (rank_offset * pawn_direction)
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    square = chess.square(target_file, target_rank)
                    piece = board.piece_at(square)
                    
                    if piece and piece.piece_type == chess.PAWN and piece.color == board.turn:
                        shield_pawns += 1
                        score += 10 if rank_offset == 1 else 6
                    elif rank_offset == 1:
                        shield_holes += 1
        
        if shield_pawns == 0:
            score -= 50
        elif shield_holes >= 2:
            score -= 25
        
        enemy_queens = board.pieces(chess.QUEEN, not board.turn)
        for queen_sq in enemy_queens:
            q_file = chess.square_file(queen_sq)
            q_rank = chess.square_rank(queen_sq)
            distance = max(abs(king_file - q_file), abs(king_rank - q_rank))
            
            if distance <= 2:
                score -= 70  
            elif distance <= 4:
                score -= 35  
            elif distance <= 5:
                score -= 15  
        
        king_zone_attacks = 0
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    square = chess.square(target_file, target_rank)
                    attackers = board.attackers(not board.turn, square)
                    
                    for attacker_sq in attackers:
                        attacker = board.piece_at(attacker_sq)
                        if attacker:
                            if attacker.piece_type == chess.QUEEN:
                                king_zone_attacks += 3
                            elif attacker.piece_type == chess.ROOK:
                                king_zone_attacks += 2
                            else:
                                king_zone_attacks += 1
        
        if king_zone_attacks >= 5:
            score -= 60
        elif king_zone_attacks >= 3:
            score -= 30
        elif king_zone_attacks >= 1:
            score -= 10
        
        escape_squares = 0
        for file_offset in [-1, 0, 1]:
            for rank_offset in [-1, 0, 1]:
                if file_offset == 0 and rank_offset == 0:
                    continue
                
                target_file = king_file + file_offset
                target_rank = king_rank + rank_offset
                
                if 0 <= target_file <= 7 and 0 <= target_rank <= 7:
                    square = chess.square(target_file, target_rank)
                    piece = board.piece_at(square)
                    
                    if piece is None or piece.color != board.turn:
                        if not board.is_attacked_by(not board.turn, square):
                            escape_squares += 1
        
        if escape_squares >= 3:
            score += 15
        elif escape_squares == 0:
            score -= 25  
        
        return score
    
    def _evaluate_king_activity(self, board: chess.Board) -> int:
        """
        Endgame king activity evaluation - in endgames the king becomes a powerful
        attacking piece instead of something that needs to hide.
        
        The king should:
        - Advance toward the center (kings fight in the endgame!)
        - Support passed pawns (escort them to promotion)
        - Attack enemy pawns (eat them!)
        - Cut off the enemy king (opposition matters)
        
        This is a completely different way of thinking about the king compared to
        middlegame king safety, which is why phase detection is so important.
        Returns activity score from current player's perspective.
        """
        score = 0
        
        our_king = board.king(board.turn)
        enemy_king = board.king(not board.turn)
        
        if not our_king:
            return 0
        
        king_file = chess.square_file(our_king)
        king_rank = chess.square_rank(our_king)
        
        file_from_center = min(abs(king_file - 3), abs(king_file - 4))
        rank_from_center = min(abs(king_rank - 3), abs(king_rank - 4))
        center_distance = file_from_center + rank_from_center
        
        score += (6 - center_distance) * 15
        
        if board.turn == chess.WHITE:
            score += king_rank * 8
        else:
            score += (7 - king_rank) * 8
        
        our_passed_pawns = self._get_passed_pawn_squares(board, board.turn)
        for pawn_sq in our_passed_pawns:
            pawn_file = chess.square_file(pawn_sq)
            pawn_rank = chess.square_rank(pawn_sq)
            
            distance = max(abs(king_file - pawn_file), abs(king_rank - pawn_rank))
            
            if distance <= 2:
                score += 25
            elif distance <= 3:
                score += 12
        
        if enemy_king:
            enemy_file = chess.square_file(enemy_king)
            enemy_rank = chess.square_rank(enemy_king)
            
            king_distance = max(abs(king_file - enemy_file), abs(king_rank - enemy_rank))
            
            pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                        len(board.pieces(chess.PAWN, chess.BLACK))
            
            if pawn_count > 0 and pawn_count <= 6:
                on_same_file = (king_file == enemy_file)
                on_same_rank = (king_rank == enemy_rank)
                on_same_diagonal = (abs(king_file - enemy_file) == abs(king_rank - enemy_rank))
                
                if (on_same_file or on_same_rank or on_same_diagonal):
                    if king_distance % 2 == 1:
                        score += 20  
        
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        pawns_in_king_range = 0
        
        for pawn_sq in enemy_pawns:
            pawn_file = chess.square_file(pawn_sq)
            pawn_rank = chess.square_rank(pawn_sq)
            distance = max(abs(king_file - pawn_file), abs(king_rank - pawn_rank))
            
            if distance <= 2:
                pawns_in_king_range += 1
        
        score += pawns_in_king_range * 10
        
        return score
    
    def _evaluate_mobility(self, board: chess.Board) -> int:
        """
       Enhanced mobility evaluation with quality weighting.
        
        Not all moves are created equal! This doesn't just count moves, it weights
        them by quality:
        - Central moves weighted higher
        - Attacking moves weighted higher
        - Safe squares weighted higher
        
        The basic idea is that having more good moves available = better position,
        but we care more about meaningful moves than just random piece shuffling.
        Returns mobility advantage from current player's perspective.
        """
        our_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        enemy_moves = len(list(board.legal_moves))
        board.turn = not board.turn
        raw_mobility = our_moves - enemy_moves
        our_attacks = self._count_attacks_on_enemy_pieces(board)
        score = (raw_mobility * 3) + (our_attacks * 2)
        
        return score
    
    def _count_attacks_on_enemy_pieces(self, board: chess.Board) -> int:
        """Counts how many enemy pieces we're currently attacking.
        
        This is part of the mobility quality bonus, pieces that attack enemy pieces
        are doing useful work. It's way better to have moves that create threats than
        moves that just shuffle pieces around aimlessly."""
        attacks = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    attacks += 1
        return attacks
    
    def _evaluate_passed_pawns(self, board: chess.Board) -> int:
        """
        Comprehensive passed pawn evaluation, passed pawns are one of the most
        important positional factors, especially in endgames.
        
        A passed pawn is super valuable because it can potentially promote to a queen
        if it advances far enough. This function considers:
        - Advancement rank (further advanced = way more dangerous)
        - Support from own pieces, especially rooks behind it (Tarrasch principle)
        - Distance from both kings (far from enemy king = unstoppable)
        - Whether it's blocked (blockaded pawns are less scary)
        
        The values scale exponentially because a pawn on the 6th rank is WAY more
        dangerous than one on the 3rd rank. Returns passed pawn value from current
        player's perspective.
        """
        score = 0
        
        our_pawns = board.pieces(chess.PAWN, board.turn)
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        
        our_king = board.king(board.turn)
        enemy_king = board.king(not board.turn)
        
        for pawn_square in our_pawns:
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            
            is_passed = True
            for enemy_pawn_square in enemy_pawns:
                enemy_file = chess.square_file(enemy_pawn_square)
                enemy_rank = chess.square_rank(enemy_pawn_square)
                
                if abs(enemy_file - file) <= 1:
                    if board.turn == chess.WHITE:
                        if enemy_rank > rank:
                            is_passed = False
                            break
                    else:
                        if enemy_rank < rank:
                            is_passed = False
                            break
            
            if not is_passed:
                continue
            
            advancement = rank if board.turn == chess.WHITE else (7 - rank)
            
            passed_values = {
                0: 0, 1: 0, 2: 10, 3: 20, 4: 40, 5: 70, 6: 120, 7: 0 
            }
            
            bonus = passed_values.get(advancement, 0)
            
            pawn_defenders = 0
            for defender_sq in board.attackers(board.turn, pawn_square):
                defender = board.piece_at(defender_sq)
                if defender and defender.piece_type == chess.PAWN:
                    pawn_defenders += 1
            
            if pawn_defenders > 0:
                bonus = int(bonus * 1.4)  
            
            phase = self.get_game_phase(board)
            if phase < 0.4:  
                if our_king:
                    our_king_file = chess.square_file(our_king)
                    our_king_rank = chess.square_rank(our_king)
                    our_king_dist = max(abs(our_king_file - file), 
                                       abs(our_king_rank - rank))
                    
                    if our_king_dist <= 2:
                        bonus += 15
                
                if enemy_king:
                    enemy_king_file = chess.square_file(enemy_king)
                    enemy_king_rank = chess.square_rank(enemy_king)
                    enemy_king_dist = max(abs(enemy_king_file - file), 
                                         abs(enemy_king_rank - rank))
                    
                    if enemy_king_dist >= 4:
                        bonus += 25
                    elif enemy_king_dist >= 3:
                        bonus += 12
            
            our_rooks = board.pieces(chess.ROOK, board.turn)
            for rook_sq in our_rooks:
                rook_file = chess.square_file(rook_sq)
                rook_rank = chess.square_rank(rook_sq)
                
                if rook_file == file:
                    if board.turn == chess.WHITE and rook_rank < rank:
                        bonus += 20
                    elif board.turn == chess.BLACK and rook_rank > rank:
                        bonus += 20
            
            advance_direction = 1 if board.turn == chess.WHITE else -1
            front_square = chess.square(file, rank + advance_direction)
            
            if 0 <= rank + advance_direction <= 7:
                blocker = board.piece_at(front_square)
                if blocker and blocker.color != board.turn:
                    if blocker.piece_type == chess.KNIGHT:
                        bonus = int(bonus * 0.6)  
                    elif blocker.piece_type in [chess.BISHOP, chess.ROOK]:
                        bonus = int(bonus * 0.7)
                    else:
                        bonus = int(bonus * 0.8)
            
            score += bonus
        
        return score
    
    def _get_passed_pawn_squares(self, board: chess.Board, color: chess.Color) -> List[int]:
        """Helper function that returns a list of squares where we have passed pawns.
        
        A pawn is passed if there are no enemy pawns on the same file or adjacent
        files ahead of it. This is used by multiple evaluation functions that need
        to know where the passed pawns are."""
        passed_pawns = []
        our_pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        
        for pawn_square in our_pawns:
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            
            is_passed = True
            for enemy_pawn_square in enemy_pawns:
                enemy_file = chess.square_file(enemy_pawn_square)
                enemy_rank = chess.square_rank(enemy_pawn_square)
                
                if abs(enemy_file - file) <= 1:
                    if color == chess.WHITE:
                        if enemy_rank > rank:
                            is_passed = False
                            break
                    else:
                        if enemy_rank < rank:
                            is_passed = False
                            break
            
            if is_passed:
                passed_pawns.append(pawn_square)
        
        return passed_pawns
    
    def _evaluate_bishop_pair(self, board: chess.Board) -> int:
        """
        Advanced bishop pair evaluation with positional context.
        
        The bishop pair (having both bishops while the opponent doesn't) is a well-known
        advantage, but its strength depends heavily on the position. This function is
        sophisticated and considers:
        - Board openness (bishops love open positions, hate closed ones)
        - Game phase (more valuable in endgames)
        - Bishop mobility (useless bishops stuck behind pawns don't get the bonus)
        - Color complex weaknesses (if opponent has only one bishop, their weak squares matter)
        
        I made this really detailed because the bishop pair is one of those positional
        factors that varies wildly in value depending on context. Returns bishop pair
        advantage from current player's perspective.
        """
        score = 0
        color = board.turn
        opponent = not color
        
        our_bishops = list(board.pieces(chess.BISHOP, color))
        enemy_bishops = list(board.pieces(chess.BISHOP, opponent))
        
        our_bishop_count = len(our_bishops)
        enemy_bishop_count = len(enemy_bishops)
        
        if our_bishop_count >= 2:
            base_bonus = 50
            
            openness_factor = self._calculate_board_openness(board)
            
            if openness_factor < 0.6:
                base_bonus = int(base_bonus * 0.5)
            elif openness_factor < 1.0:
                base_bonus = int(base_bonus * 0.7)
            elif openness_factor > 1.4:
                base_bonus = int(base_bonus * 1.4)
            elif openness_factor > 1.2:
                base_bonus = int(base_bonus * 1.2)
            
            phase_multiplier = self._get_bishop_pair_phase_multiplier(board)
            base_bonus = int(base_bonus * phase_multiplier)
            
            our_bishop_mobility = self._calculate_bishop_mobility(board, our_bishops, color)
            avg_mobility = our_bishop_mobility / len(our_bishops) if our_bishops else 0
            
            if avg_mobility < 4:
                base_bonus = int(base_bonus * 0.3)
            elif avg_mobility < 7:
                base_bonus = int(base_bonus * 0.7)
            elif avg_mobility > 12:
                base_bonus = int(base_bonus * 1.15)
            
            our_queen = board.pieces(chess.QUEEN, color)
            enemy_queen = board.pieces(chess.QUEEN, opponent)
            
            if len(our_queen) > 0 and len(enemy_queen) > 0:
                base_bonus += 10
            
            total_pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                         len(board.pieces(chess.PAWN, chess.BLACK))
            
            if total_pawns >= 14:
                base_bonus += 5
            
            score += base_bonus
        
        if enemy_bishop_count >= 2:
            base_penalty = 50
            
            openness_factor = self._calculate_board_openness(board)
            
            if openness_factor < 0.6:
                base_penalty = int(base_penalty * 0.5)
            elif openness_factor < 1.0:
                base_penalty = int(base_penalty * 0.7)
            elif openness_factor > 1.4:
                base_penalty = int(base_penalty * 1.4)
            elif openness_factor > 1.2:
                base_penalty = int(base_penalty * 1.2)
            
            phase_multiplier = self._get_bishop_pair_phase_multiplier(board)
            base_penalty = int(base_penalty * phase_multiplier)
            
            enemy_bishop_mobility = self._calculate_bishop_mobility(board, enemy_bishops, opponent)
            avg_enemy_mobility = enemy_bishop_mobility / len(enemy_bishops) if enemy_bishops else 0
            
            if avg_enemy_mobility < 4:
                base_penalty = int(base_penalty * 0.3)
            elif avg_enemy_mobility < 7:
                base_penalty = int(base_penalty * 0.7)
            elif avg_enemy_mobility > 12:
                base_penalty = int(base_penalty * 1.15)
            
            our_queen = board.pieces(chess.QUEEN, color)
            enemy_queen = board.pieces(chess.QUEEN, opponent)
            
            if len(our_queen) > 0 and len(enemy_queen) > 0:
                base_penalty += 10
            
            score -= base_penalty
        
        if enemy_bishop_count == 1 and our_bishop_count >= 1:
            color_complex_bonus = 20
            
            phase_multiplier = self._get_bishop_pair_phase_multiplier(board)
            if phase_multiplier > 1.2:
                color_complex_bonus = int(color_complex_bonus * 1.3)
            
            score += color_complex_bonus
        
        if our_bishop_count == 1 and enemy_bishop_count >= 1:
            color_complex_penalty = 20
            
            phase_multiplier = self._get_bishop_pair_phase_multiplier(board)
            if phase_multiplier > 1.2:
                color_complex_penalty = int(color_complex_penalty * 1.3)
            
            score -= color_complex_penalty
        
        return score
    
    def _calculate_board_openness(self, board: chess.Board) -> float:
        """Calculates how open the board is on a scale from 0.0 (closed) to 2.0 (very open).
        
        Open positions favor long-range pieces like bishops and rooks. Closed positions
        favor knights. This looks at:
        - Number of open files (no pawns on them)
        - Locked pawn chains (pawns blocking each other)
        - Central pawn configuration
        - Total pawn count
        
        This metric is used by bishop pair evaluation and other positional factors that
        care about board structure."""
        openness_score = 1.0
        
        open_files = 0
        for file in range(8):
            has_pawn = False
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    has_pawn = True
                    break
            if not has_pawn:
                open_files += 1
        
        openness_score += (open_files * 0.15)
        
        locked_pawns = self._count_locked_pawns(board)
        openness_score -= (locked_pawns * 0.08)
        
        central_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        central_pawns = sum(1 for sq in central_squares 
                           if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN)
        
        if central_pawns >= 3:
            openness_score -= 0.3
        elif central_pawns <= 1:
            openness_score += 0.2
        
        total_pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + \
                     len(board.pieces(chess.PAWN, chess.BLACK))
        
        if total_pawns <= 10:
            openness_score += 0.3
        elif total_pawns >= 15:
            openness_score -= 0.2
        
        return max(0.3, min(2.0, openness_score))
    
    def _count_locked_pawns(self, board: chess.Board) -> int:
        """Counts pawns that are locked face-to-face with enemy pawns.
        
        Locked pawns create closed positions because they can't be easily removed.
        This is relevant for openness calculations and understanding the strategic
        nature of the position."""
        locked_count = 0
        
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        for pawn_square in white_pawns:
            rank = chess.square_rank(pawn_square)
            file = chess.square_file(pawn_square)
            
            if rank < 7:
                ahead_square = chess.square(file, rank + 1)
                ahead_piece = board.piece_at(ahead_square)
                
                if ahead_piece and ahead_piece.piece_type == chess.PAWN and \
                   ahead_piece.color == chess.BLACK:
                    locked_count += 1
        
        return locked_count
    
    def _calculate_bishop_mobility(self, board: chess.Board, bishops: list, 
                                   color: chess.Color) -> int:
        """ Calculates total legal moves available to bishops.
        
        This is used by bishop pair evaluation to make sure the bishops are actually
        active. Having the bishop pair means nothing if both bishops are trapped
        behind locked pawn chains with no squares to move to."""
        total_mobility = 0
        
        for bishop_square in bishops:
            moves = board.attacks(bishop_square)
            
            for target_square in moves:
                target_piece = board.piece_at(target_square)
                if target_piece is None or target_piece.color != color:
                    total_mobility += 1
        
        return total_mobility
    
    def _get_bishop_pair_phase_multiplier(self, board: chess.Board) -> float:
        """Phase multiplier for bishop pair value: 0.7x in opening to 1.5x in endgame.
        
        Bishop pairs get more valuable as the game goes on because pawns get traded
        off and the board opens up, giving the bishops more scope. This multiplier
        adjusts the bishop pair bonus based on game phase."""
        move_number = board.fullmove_number
        
        piece_count = sum(len(board.pieces(pt, c)) 
                         for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
                         for c in [chess.WHITE, chess.BLACK])
        
        if piece_count <= 6 or move_number > 40:
            return 1.5
        elif move_number <= 12 and piece_count >= 16:
            return 0.7
        else:
            phase_progress = (16 - piece_count) / 10.0
            phase_progress = max(0.0, min(1.0, phase_progress))
            return 0.7 + (phase_progress * 0.8)
    
    def _evaluate_development(self, board: chess.Board) -> int:
        """
        Advanced development evaluation with castling priority.
        
        Development is all about getting your pieces into the game quickly in the
        opening. This is most important early on and fades in relevance as the game
        progresses. The function looks at:
        - Castling (super important for safety and rook connection)
        - Minor piece development off the back rank
        - Rook placement on open files
        - Premature queen development (bringing the queen out too early is risky)
        - Relative development compared to opponent
        
        Returns development score from current player's perspective, weighted by
        game phase so it only matters when it should.
        """
        score = 0
        phase = self.get_game_phase(board)
        
        if phase < 0.15:
            return 0
        
        color = board.turn
        opponent = not color
        back_rank = 0 if color == chess.WHITE else 7
        
        if board.has_castling_rights(color):
            if board.fullmove_number > 10:
                score -= int(20 * phase)
        else:
            king_square = board.king(color)
            if king_square:
                king_file = chess.square_file(king_square)
                
                if king_file in [2, 6]:
                    score += int(60 * phase)
                    
                    rooks = board.pieces(chess.ROOK, color)
                    if len(rooks) == 2:
                        rook_squares = list(rooks)
                        if self._are_rooks_connected(board, rook_squares[0], rook_squares[1]):
                            score += int(25 * phase)
        
        developed_minor_pieces = 0
        
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            pieces = board.pieces(piece_type, color)
            
            for square in pieces:
                rank = chess.square_rank(square)
                
                if rank == back_rank:
                    if board.fullmove_number > 8:
                        score -= int(15 * phase)
                    continue
                
                developed_minor_pieces += 1
                base_bonus = 15
                
                quality_bonus = self._evaluate_piece_placement_quality(
                    board, piece_type, square, color, phase
                )
                
                score += int((base_bonus + quality_bonus) * phase)
        
        rooks = board.pieces(chess.ROOK, color)
        for rook_square in rooks:
            rook_file = chess.square_file(rook_square)
            rook_rank = chess.square_rank(rook_square)
            
            if self._is_open_file(board, rook_file):
                score += int(30 * phase)
            elif self._is_semiopen_file(board, rook_file, color):
                score += int(15 * phase)
            
            seventh_rank = 6 if color == chess.WHITE else 1
            if rook_rank == seventh_rank:
                score += int(35 * (phase * 0.7 + 0.3))
        
        queens = board.pieces(chess.QUEEN, color)
        queen_square = list(queens)[0] if queens else None
        if queen_square is not None:
            queen_rank = chess.square_rank(queen_square)
            
            if queen_rank != back_rank and board.fullmove_number <= 10:
                undeveloped_minors = sum(1 for pt in [chess.KNIGHT, chess.BISHOP]
                                        for sq in board.pieces(pt, color)
                                        if chess.square_rank(sq) == back_rank)
                
                if undeveloped_minors >= 2:
                    score -= int(30 * phase)
                elif undeveloped_minors == 1:
                    score -= int(15 * phase)
        
        if phase > 0.5:
            opponent_back_rank = 0 if opponent == chess.WHITE else 7
            opponent_developed_minors = sum(1 for pt in [chess.KNIGHT, chess.BISHOP]
                                           for sq in board.pieces(pt, opponent)
                                           if chess.square_rank(sq) != opponent_back_rank)
            
            development_difference = developed_minor_pieces - opponent_developed_minors
            if development_difference > 0:
                score += int(development_difference * 10 * phase)
        
        return score
    
    def _evaluate_piece_placement_quality(self, board, piece_type, square, color, phase):
        """Evaluates how good a piece's placement is beyond just the PST values.
        
        This gives bonuses for things like:
        - Knights on outposts (protected advanced squares)
        - Bishops on fianchetto positions or long diagonals
        - Central placement
        - Penalizes rim knights (a knight on the rim is dim!)
        
        Returns a quality score from -12 (terrible placement) to +20 (excellent placement)."""
        quality = 0
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if file in [2, 3, 4, 5]:
            quality += 8
        elif file in [0, 7]:
            quality -= 5
        
        if piece_type == chess.KNIGHT:
            if file in [0, 7] or rank in [0, 7]:
                quality -= 12
            if file in [3, 4] and rank in [3, 4]:
                quality += 15
            if self._is_outpost(board, square, color):
                quality += 20
        
        elif piece_type == chess.BISHOP:
            if self._is_fianchetto_position(board, square, color):
                quality += 10
            if self._controls_long_diagonal(board, square):
                quality += 12
        
        return quality
    
    def _are_rooks_connected(self, board, rook1_square, rook2_square):
        """Checks if two rooks can see each other on the same rank or file.
        
        Connected rooks are powerful because they defend each other and can coordinate
        attacks. This checks that they're on the same rank/file and that there are no
        pieces blocking the path between them."""
        if chess.square_rank(rook1_square) != chess.square_rank(rook2_square):
            return False
        
        min_file = min(chess.square_file(rook1_square), chess.square_file(rook2_square))
        max_file = max(chess.square_file(rook1_square), chess.square_file(rook2_square))
        rank = chess.square_rank(rook1_square)
        
        for file in range(min_file + 1, max_file):
            square = chess.square(file, rank)
            if board.piece_at(square) is not None:
                return False
        
        return True
    
    def _is_open_file(self, board, file):
        """Checks if a file has no pawns on it at all.
        
        Open files are highways for rooks - they can slide up and down freely. Rooks
        love open files and should be placed on them when possible."""
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                return False
        return True
    
    def _is_semiopen_file(self, board, file, color):
        """Checks if a file has no pawns of our color on it.
        
        Semi-open files are still good for rooks even if the opponent has a pawn there,
        because our rook can attack that pawn. Not as good as fully open files but
        still valuable."""
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                return False
        return True
    
    def _is_outpost(self, board, square, color):
        """Checks if a square is a protected outpost for our pieces.
        
        An outpost is an advanced square that:
        - Is protected by our pawn
        - Cannot be easily attacked by enemy pawns
        - Is in enemy territory
        
        Pieces (especially knights) on outposts are super strong because they're
        stable and can't be kicked out easily."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if (color == chess.WHITE and rank < 4) or (color == chess.BLACK and rank > 3):
            return False
        
        pawn_direction = 1 if color == chess.WHITE else -1
        support_rank = rank - pawn_direction
        
        for support_file in [file - 1, file + 1]:
            if 0 <= support_file <= 7:
                support_square = chess.square(support_file, support_rank)
                piece = board.piece_at(support_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    if not self._can_challenge_with_pawn(board, square, not color):
                        return True
        
        return False
    
    def _can_challenge_with_pawn(self, board, square, color):
        """Checks if the given color can attack a square with a pawn.
        
        This is used by outpost detection, if enemy pawns can advance to attack
        our piece, then it's not a true outpost because it can be challenged and
        kicked out."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        for pawn_file in [file - 1, file + 1]:
            if 0 <= pawn_file <= 7:
                direction = 1 if color == chess.WHITE else -1
                for distance in range(1, 4):
                    check_rank = rank - (direction * distance)
                    if 0 <= check_rank <= 7:
                        check_square = chess.square(pawn_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            return True
        return False
    
    def _is_fianchetto_position(self, board, square, color):
        """Checks if a bishop is in a fianchetto position (on g2/b2 for white, g7/b7 for black).
        
        Fianchettoed bishops are a specific setup where the bishop controls a long diagonal
        from behind a pawn chain. This is a classic positional pattern that's often strong."""
        fianchetto_squares = {
            chess.WHITE: [chess.B2, chess.G2],
            chess.BLACK: [chess.B7, chess.G7]
        }
        return square in fianchetto_squares.get(color, [])
    
    def _controls_long_diagonal(self, board, square):
        """Checks if a bishop is positioned on one of the long diagonals (a1-h8 or a8-h1).
        
        Bishops on long diagonals have maximum scope and can influence the entire board.
        This is generally a good placement for bishops."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        return file == rank or file + rank == 7
    
    def _evaluate_center_control(self, board: chess.Board) -> int:
        """
        Enhanced center control evaluation, controlling the center is fundamental chess strategy.
        
        The center (d4, d5, e4, e5) is the most important part of the board because:
        - Pieces in the center control more squares
        - Central pawns restrict enemy pieces
        - Center control allows flexibility in choosing which side to attack
        
        This function looks at:
        - Attack count on center squares (weighted by piece type - pawn control is best)
        - Actual occupancy of center squares (having pieces there is even better)
        - Extended center control (supporting squares around the center)
        
        Returns center control advantage from current player's perspective.
        """
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.F4,
            chess.C5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6
        ]
        
        score = 0
        
        for square in center_squares:
            our_attackers = board.attackers(board.turn, square)
            enemy_attackers = board.attackers(not board.turn, square)
            
            our_attack_value = 0
            for attacker_sq in our_attackers:
                attacker = board.piece_at(attacker_sq)
                if attacker:
                    if attacker.piece_type == chess.PAWN:
                        our_attack_value += 7  
                    elif attacker.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        our_attack_value += 5
                    else:
                        our_attack_value += 3
            
            enemy_attack_value = 0
            for attacker_sq in enemy_attackers:
                attacker = board.piece_at(attacker_sq)
                if attacker:
                    if attacker.piece_type == chess.PAWN:
                        enemy_attack_value += 7
                    elif attacker.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        enemy_attack_value += 5
                    else:
                        enemy_attack_value += 3
            
            score += our_attack_value - enemy_attack_value
            
            piece = board.piece_at(square)
            if piece:
                if piece.color == board.turn:
                    if piece.piece_type == chess.PAWN:
                        score += 20  
                    elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        score += 12  
                else:
                    if piece.piece_type == chess.PAWN:
                        score -= 20
                    elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        score -= 12
        
        for square in extended_center:
            piece = board.piece_at(square)
            if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if piece.color == board.turn:
                    score += 3  
                else:
                    score -= 3
        
        return score
    
    def _evaluate_rook_open_files(self, board: chess.Board) -> int:
        """
        Advanced rook placement evaluation, rooks need open files to be effective.
        
        This looks at multiple aspects of rook quality:
        - Open files (no pawns) - rooks love these
        - Semi-open files (no our pawns) - still good
        - 7th rank presence (rooks on the 7th rank are devastating)
        - Doubled rooks on files (two rooks supporting each other)
        - Battery formation (rook + queen on same file = very dangerous)
        
        Rooks are weird pieces that are weak in closed positions but monsters on
        open files, so proper placement makes a huge difference. Returns rook
        placement quality from current player's perspective.
        """
        our_rooks = list(board.pieces(chess.ROOK, board.turn))
        our_pawns = board.pieces(chess.PAWN, board.turn)
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        
        score = 0
        rook_files = {}  
        
        for rook_sq in our_rooks:
            file = chess.square_file(rook_sq)
            rank = chess.square_rank(rook_sq)
            
            if file not in rook_files:
                rook_files[file] = []
            rook_files[file].append(rook_sq)
            
            our_pawn_on_file = any(chess.square_file(p) == file for p in our_pawns)
            enemy_pawn_on_file = any(chess.square_file(p) == file for p in enemy_pawns)
            
            if not our_pawn_on_file and not enemy_pawn_on_file:
                score += 35  
            elif not our_pawn_on_file:
                score += 20  
            
            seventh_rank = 6 if board.turn == chess.WHITE else 1
            if rank == seventh_rank:
                score += 40  
                
                enemy_king = board.king(not board.turn)
                if enemy_king:
                    enemy_king_rank = chess.square_rank(enemy_king)
                    back_rank = 7 if board.turn == chess.WHITE else 0
                    if enemy_king_rank == back_rank:
                        score += 20  
        
        for file, rooks_on_file in rook_files.items():
            if len(rooks_on_file) >= 2:
                if len(rooks_on_file) == 2:
                    if self._are_rooks_connected(board, rooks_on_file[0], rooks_on_file[1]):
                        score += 25  
        
        queens = board.pieces(chess.QUEEN, board.turn)
        our_queen = list(queens)[0] if queens else None
        if our_queen:
            queen_file = chess.square_file(our_queen)
            if queen_file in rook_files:
                score += 30  
        
        return score
    
    def _evaluate_pawn_structure(self, board: chess.Board) -> int:
        """
        Comprehensive pawn structure evaluation, pawn structure is permanent!
        
        Unlike pieces which can move around, pawn structure tends to be permanent for
        long stretches of the game. Bad pawn structure causes long-term weaknesses.
        
        Penalties for:
        - Doubled pawns (multiple pawns stacked on same file - weak and immobile)
        - Isolated pawns (no friendly pawns on adjacent files - vulnerable)
        - Backward pawns (behind neighbors and can't advance safely)
        - Pawn islands (disconnected pawn groups - harder to defend)
        
        Bonuses for:
        - Pawn chains (connected pawns supporting each other diagonally)
        - Pawn phalanx (pawns side-by-side - strong and flexible)
        
        I spent a lot of time getting these penalties right because pawn structure
        evaluation is subtle but really important. Returns structure quality from
        current player's perspective.
        """
        our_pawns = list(board.pieces(chess.PAWN, board.turn))
        score = 0
        
        file_counts = [0] * 8
        pawn_ranks_by_file = {i: [] for i in range(8)}
        
        for pawn_sq in our_pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            file_counts[file] += 1
            pawn_ranks_by_file[file].append(rank)
        
        for count in file_counts:
            if count >= 2:
                score -= (count - 1) * 18
                if count >= 3:
                    score -= 25 
        
        for file in range(8):
            if file_counts[file] > 0:
                has_neighbor = False
                if file > 0 and file_counts[file - 1] > 0:
                    has_neighbor = True
                if file < 7 and file_counts[file + 1] > 0:
                    has_neighbor = True
                
                if not has_neighbor:
                    score -= 25
                    
                    phase = self.get_game_phase(board)
                    if phase < 0.3:
                        score -= 15
        
        for pawn_sq in our_pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            is_backward = False
            
            for neighbor_file in [file - 1, file + 1]:
                if 0 <= neighbor_file <= 7 and file_counts[neighbor_file] > 0:
                    neighbor_ranks = pawn_ranks_by_file[neighbor_file]
                    
                    if board.turn == chess.WHITE:
                        if rank < min(neighbor_ranks):
                            if rank < 7:
                                advance_sq = chess.square(file, rank + 1)
                                if board.is_attacked_by(not board.turn, advance_sq):
                                    is_backward = True
                    else:
                        if rank > max(neighbor_ranks):
                            if rank > 0:
                                advance_sq = chess.square(file, rank - 1)
                                if board.is_attacked_by(not board.turn, advance_sq):
                                    is_backward = True
            
            if is_backward:
                score -= 20
       
        islands = 0
        in_island = False
        
        for file in range(8):
            if file_counts[file] > 0:
                if not in_island:
                    islands += 1
                    in_island = True
            else:
                in_island = False
        
        if islands >= 4:
            score -= 40
        elif islands >= 3:
            score -= 20
        elif islands >= 2:
            score -= 8
        
        pawn_chain_count = 0
        for pawn_sq in our_pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            pawn_direction = 1 if board.turn == chess.WHITE else -1
            behind_rank = rank - pawn_direction
            
            if 0 <= behind_rank <= 7:
                for side_file in [file - 1, file + 1]:
                    if 0 <= side_file <= 7:
                        behind_sq = chess.square(side_file, behind_rank)
                        behind_piece = board.piece_at(behind_sq)
                        if behind_piece and behind_piece.piece_type == chess.PAWN and \
                           behind_piece.color == board.turn:
                            pawn_chain_count += 1
                            break
        
        score += pawn_chain_count * 5
        
        phalanx_count = 0
        for pawn_sq in our_pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            if file < 7:
                right_sq = chess.square(file + 1, rank)
                right_piece = board.piece_at(right_sq)
                if right_piece and right_piece.piece_type == chess.PAWN and \
                   right_piece.color == board.turn:
                    phalanx_count += 1
        
        score += phalanx_count * 8
        
        return score
    
    def _evaluate_knight_outposts(self, board: chess.Board) -> int:
        """
        Knight outpost evaluation, knights on strong outposts dominate games.
        
        An outpost is a square that:
        - Is in enemy territory (5th rank or beyond)
        - Is protected by our pawn (so the knight can't be kicked out)
        - Cannot be easily challenged by enemy pawns (permanent stability)
        
        Knights on outposts are incredibly strong because they're stable, active,
        and deep in enemy territory where they create constant threats. The closer
        to the center and the deeper into enemy territory, the better.
        
        Returns outpost quality from current player's perspective.

        """
        our_knights = board.pieces(chess.KNIGHT, board.turn)
        our_pawns = board.pieces(chess.PAWN, board.turn)
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        
        score = 0
        
        for knight_sq in our_knights:
            rank = chess.square_rank(knight_sq)
            file = chess.square_file(knight_sq)
            
            if board.turn == chess.WHITE:
                if rank < 4:  
                    continue
            else:
                if rank > 3:  
                    continue
            
            defended_by_pawn = False
            for pawn_sq in our_pawns:
                pawn_file = chess.square_file(pawn_sq)
                pawn_rank = chess.square_rank(pawn_sq)
                
                if abs(pawn_file - file) == 1:
                    if board.turn == chess.WHITE and pawn_rank == rank - 1:
                        defended_by_pawn = True
                        break
                    elif board.turn == chess.BLACK and pawn_rank == rank + 1:
                        defended_by_pawn = True
                        break
            
            if not defended_by_pawn:
                continue
            
            can_be_challenged = False
            for enemy_pawn_sq in enemy_pawns:
                enemy_file = chess.square_file(enemy_pawn_sq)
                enemy_rank = chess.square_rank(enemy_pawn_sq)
                
                if abs(enemy_file - file) == 1:
                    if board.turn == chess.WHITE:
                        if enemy_rank < rank:
                            can_be_challenged = True
                            break
                    else:
                        if enemy_rank > rank:
                            can_be_challenged = True
                            break
            
            if not can_be_challenged:
                base_bonus = 35
                
                if board.turn == chess.WHITE:
                    if rank >= 5:
                        base_bonus += 15  
                else:
                    if rank <= 2:
                        base_bonus += 15  
                
                if file in [3, 4]:
                    base_bonus += 10
                
                score += base_bonus
        
        return score
    
    def _evaluate_connected_rooks(self, board: chess.Board) -> int:
        """
        Connected rooks evaluation, rooks are stronger when they work together.
        
        Rooks are connected when they defend each other on the same rank or file
        with no pieces between them. This is a strong positional advantage because:
        - They defend each other
        - They can coordinate attacks
        - Doubling on files/ranks creates major threats
        
        Returns bonus if rooks are connected, zero otherwise.
        """
        if self._has_connected_rooks(board):
            return 25
        return 0
    
    def _count_development(self, board: chess.Board) -> int:
        """Counts developed minor pieces (off the back rank).
        
        This is a simple helper for feature extraction that counts how many knights
        and bishops have moved off their starting squares. Used by ML training to
        quantify development numerically."""
        developed = 0
        back_rank = 0 if board.turn == chess.WHITE else 7
        
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            pieces = board.pieces(piece_type, board.turn)
            for square in pieces:
                if chess.square_rank(square) != back_rank:
                    developed += 1
        
        return developed
    
    def _count_center_attacks(self, board: chess.Board) -> int:
        """Counts attacks on center squares for feature extraction.
        
        Simple count of how many times we attack center squares versus how many times
        the opponent does. Used by ML training to quantify center control numerically."""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        our_attacks = sum(len(board.attackers(board.turn, sq)) for sq in center_squares)
        enemy_attacks = sum(len(board.attackers(not board.turn, sq)) for sq in center_squares)
        
        return our_attacks - enemy_attacks
    
    def _count_rooks_on_open_files(self, board: chess.Board) -> int:
        """Counts rooks on open or semi-open files for feature extraction.
        
        Gives 2 points for rooks on fully open files, 1 point for semi-open files.
        Used by ML training to quantify rook placement quality numerically."""
        our_rooks = board.pieces(chess.ROOK, board.turn)
        our_pawns = board.pieces(chess.PAWN, board.turn)
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        
        rook_count = 0
        
        for rook_sq in our_rooks:
            file = chess.square_file(rook_sq)
            
            our_pawn_on_file = any(chess.square_file(p) == file for p in our_pawns)
            enemy_pawn_on_file = any(chess.square_file(p) == file for p in enemy_pawns)
            
            if not our_pawn_on_file and not enemy_pawn_on_file:
                rook_count += 2  
            elif not our_pawn_on_file:
                rook_count += 1  
        
        return rook_count
    
    def _evaluate_pawn_structure_simple(self, board: chess.Board) -> int:
        """Simplified pawn structure evaluation for feature extraction.
        
        This is a stripped-down version that just counts doubled pawns and isolated
        pawns without all the complexity of the full evaluation. Used by ML training
        to provide a simple numerical feature."""
        our_pawns = board.pieces(chess.PAWN, board.turn)
        penalty = 0
        
        file_counts = [0] * 8
        for pawn_sq in our_pawns:
            file = chess.square_file(pawn_sq)
            file_counts[file] += 1
        
        for count in file_counts:
            if count >= 2:
                penalty -= (count - 1)
        
        for file in range(8):
            if file_counts[file] > 0:
                has_neighbor = ((file > 0 and file_counts[file - 1] > 0) or
                              (file < 7 and file_counts[file + 1] > 0))
                if not has_neighbor:
                    penalty -= 1
        
        return penalty
    
    def _count_knight_outposts(self, board: chess.Board) -> int:
        """Counts knight outposts for feature extraction.
        
        Simple count of how many knights are on true outpost squares (protected by
        pawns, in enemy territory, can't be challenged). Used by ML training to
        quantify outpost strength numerically."""
        our_knights = board.pieces(chess.KNIGHT, board.turn)
        our_pawns = board.pieces(chess.PAWN, board.turn)
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        
        outposts = 0
        
        for knight_sq in our_knights:
            rank = chess.square_rank(knight_sq)
            file = chess.square_file(knight_sq)
            
            if board.turn == chess.WHITE and rank < 4:
                continue
            if board.turn == chess.BLACK and rank > 3:
                continue
            
            defended_by_pawn = any(
                abs(chess.square_file(p) - file) == 1 and
                ((board.turn == chess.WHITE and chess.square_rank(p) == rank - 1) or
                 (board.turn == chess.BLACK and chess.square_rank(p) == rank + 1))
                for p in our_pawns
            )
            
            if not defended_by_pawn:
                continue
            
            can_be_challenged = any(
                abs(chess.square_file(p) - file) == 1 and
                ((board.turn == chess.WHITE and chess.square_rank(p) < rank) or
                 (board.turn == chess.BLACK and chess.square_rank(p) > rank))
                for p in enemy_pawns
            )
            
            if not can_be_challenged:
                outposts += 1
        
        return outposts
    
    def _has_connected_rooks(self, board: chess.Board) -> bool:
        """Checks if we have connected rooks (can see each other).
        
        Returns True if any two of our rooks are on the same rank/file with no pieces
        between them, False otherwise. Used by multiple evaluation functions."""
        our_rooks = list(board.pieces(chess.ROOK, board.turn))
        
        if len(our_rooks) < 2:
            return False
        
        for i in range(len(our_rooks)):
            for j in range(i + 1, len(our_rooks)):
                rook1 = our_rooks[i]
                rook2 = our_rooks[j]
                
                file1, rank1 = chess.square_file(rook1), chess.square_rank(rook1)
                file2, rank2 = chess.square_file(rook2), chess.square_rank(rook2)
                
                if file1 == file2 or rank1 == rank2:
                    between_squares = list(chess.SquareSet.between(rook1, rook2))
                    path_clear = all(board.piece_at(sq) is None for sq in between_squares)
                    
                    if path_clear:
                        return True
        
        return False
    
    def _count_threats(self, board: chess.Board) -> int:
        """ Counts how many enemy pieces we're attacking for feature extraction.
        
        Simple count used by ML training to quantify tactical pressure numerically.
        More threats = more active position."""
        threats = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color != board.turn:
                if board.is_attacked_by(board.turn, square):
                    threats += 1
        return threats
    
    def _count_passed_pawns_simple(self, board: chess.Board) -> int:
        """Simple passed pawn counter for feature extraction.
        
        Just counts how many passed pawns we have without considering their quality,
        advancement, or support. Used by ML training as a basic numerical feature."""
        passed = 0
        our_pawns = board.pieces(chess.PAWN, board.turn)
        enemy_pawns = board.pieces(chess.PAWN, not board.turn)
        
        for pawn_sq in our_pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            is_passed = True
            for enemy_sq in enemy_pawns:
                enemy_file = chess.square_file(enemy_sq)
                enemy_rank = chess.square_rank(enemy_sq)
                
                if abs(enemy_file - file) <= 1:
                    if board.turn == chess.WHITE and enemy_rank > rank:
                        is_passed = False
                        break
                    elif board.turn == chess.BLACK and enemy_rank < rank:
                        is_passed = False
                        break
            
            if is_passed:
                passed += 1
        
        return passed
   
    def clear_cache(self) -> None:
        """Clears the evaluation cache and resets statistics.
        
        I call this between games to start fresh. The cache is useful during a single
        game but can grow large, and we want clean data for each game."""
        self.eval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_hit_rate(self) -> float:
        """Returns the cache hit rate as a percentage.
        
        This tells me how often positions are being found in the cache versus needing
        to be recalculated. Higher hit rates = better performance. I use this to tune
        the cache size and see if caching is actually helping.
        """
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def debug_evaluate(self, board: chess.Board) -> None:
        """
        Debug evaluation breakdown showing all 12 components individually.
        
        This is super useful for understanding WHY the engine evaluates a position
        a certain way. It breaks down the total evaluation into all 12 components
        so you can see exactly where the score is coming from.
        
        I use this constantly when debugging evaluation issues or trying to understand
        why my bot made a certain move. Prints a nice formatted breakdown with all
        the component scores and their weights.
        """
        print("\n" + "="*70)
        print("DEBUG EVALUATION BREAKDOWN (12 Components)")
        print("="*70)
        
        phase = self.get_game_phase(board)
        is_endgame = phase < 0.3
        
        print(f"Game Phase: {phase:.2f} ({'Opening' if phase > 0.7 else 'Endgame' if phase < 0.3 else 'Middlegame'})")
        print("-"*70)
        
        mat_pst = self._evaluate_material_and_pst(board, phase) * self.weights['material_pst_weight']
        safety = self._evaluate_piece_safety(board) * self.weights['piece_safety_weight']
        passed = self._evaluate_passed_pawns(board) * self.weights['passed_pawn_weight']
        bishop_pair = self._evaluate_bishop_pair(board) * self.weights['bishop_pair_weight']
        development = self._evaluate_development(board) * self.weights['development_weight']
        center = self._evaluate_center_control(board) * self.weights['center_control_weight']
        rook_files = self._evaluate_rook_open_files(board) * self.weights['rook_open_file_weight']
        pawn_struct = self._evaluate_pawn_structure(board) * self.weights['pawn_structure_weight']
        knight_out = self._evaluate_knight_outposts(board) * self.weights['knight_outpost_weight']
        conn_rooks = self._evaluate_connected_rooks(board) * self.weights['connected_rooks_weight']
        
        if is_endgame:
            king = self._evaluate_king_activity(board)
        else:
            king = self._evaluate_king_safety(board) * self.weights['king_safety_weight']
        
        mobility = self._evaluate_mobility(board) * self.weights['mobility_weight']
        
        total = mat_pst + safety + passed + bishop_pair + development + center + \
                rook_files + pawn_struct + knight_out + conn_rooks + king + mobility
        
        print(f"Material + PST:      {int(mat_pst):+7d} cp  (weight: {self.weights['material_pst_weight']:.2f})")
        print(f"Piece Safety:        {int(safety):+7d} cp  (weight: {self.weights['piece_safety_weight']:.2f})")
        print(f"Passed Pawns:        {int(passed):+7d} cp  (weight: {self.weights['passed_pawn_weight']:.2f})")
        print(f"Bishop Pair:         {int(bishop_pair):+7d} cp  (weight: {self.weights['bishop_pair_weight']:.2f})")
        print(f"Development:         {int(development):+7d} cp  (weight: {self.weights['development_weight']:.2f})")
        print(f"Center Control:      {int(center):+7d} cp  (weight: {self.weights['center_control_weight']:.2f})")
        print(f"Rook Placement:      {int(rook_files):+7d} cp  (weight: {self.weights['rook_open_file_weight']:.2f})")
        print(f"Pawn Structure:      {int(pawn_struct):+7d} cp  (weight: {self.weights['pawn_structure_weight']:.2f})")
        print(f"Knight Outposts:     {int(knight_out):+7d} cp  (weight: {self.weights['knight_outpost_weight']:.2f})")
        print(f"Connected Rooks:     {int(conn_rooks):+7d} cp  (weight: {self.weights['connected_rooks_weight']:.2f})")
        print(f"King ({'Activity' if is_endgame else 'Safety'}):      {int(king):+7d} cp  (weight: {self.weights.get('king_safety_weight', 1.0):.2f})")
        print(f"Mobility:            {int(mobility):+7d} cp  (weight: {self.weights['mobility_weight']:.2f})")
        print("-" * 70)
        print(f"TOTAL EVALUATION:    {int(total):+7d} cp")
        print("="*70 + "\n")
    
    def __repr__(self) -> str:
        return f"Evaluator(12_features, GM-level, positions_evaluated={self.positions_evaluated})"