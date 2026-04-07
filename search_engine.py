"""
search_engine.py - My actual search engine used by my Chess bot
It features advanced concepts like late move reductions, principle variation search,
razoring, move count pruning, mate distance pruning, aspiration windows, adaptive null moves,
and a bunch of other fun stuff I tried to get my bot to beat the easiest version of stockfish lol.

- Author: Graeme Huntley
"""

import chess
import chess.polyglot
import time
from typing import Optional, List, Tuple, Dict
from evaluator import Evaluator


class TranspositionTable:
    """In my pointless attempt to beat stockfish I iteratively 
    built over a week lol. It does the following:
    - looks at a position
    - checks to see if it has seen this exact board state before
        - If yes and it deeply analyzed the board state before it
         will use the answer it learned previously
        - Otherwise it will think in depth about the state problem
    - It stores states so that it can remeber them after solving them
    - It gets the best move that it has liked previously
    - It gets and tracks the hitrate of the table  """
    
    EXACT = 0
    LOWER_BOUND = 1  # Alpha cutoff - the answer is at least this big
    UPPER_BOUND = 2  # Beta cutoff - the answer is at most this big
    
    def __init__(self, max_size: int = 100000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def probe(self, pos_hash: int, depth: int, alpha: float, beta: float) -> Optional[int]:
        """This is the "I have totally seen you in a bar before" function"""
        if pos_hash in self.table:
            entry = self.table[pos_hash]
            if entry['depth'] >= depth:
                score = entry['score']
                bound = entry['bound']
                
                if bound == self.EXACT:
                    self.hits += 1
                    return score
                elif bound == self.LOWER_BOUND and score >= beta:
                    self.hits += 1
                    return score
                elif bound == self.UPPER_BOUND and score <= alpha:
                    self.hits += 1
                    return score
        
        self.misses += 1
        return None
    
    def store(self, pos_hash: int, score: int, depth: int, best_move: Optional[chess.Move], bound: int) -> None:
        """This writes down the solutions so that my bot can remember them later"""
        if len(self.table) < self.max_size or pos_hash in self.table:
            self.table[pos_hash] = {
                'score': score,
                'depth': depth,
                'best_move': best_move,
                'bound': bound
            }
    
    def get_best_move(self, pos_hash: int) -> Optional[chess.Move]:
        """This is the "which move did I like last time" function"""
        if pos_hash in self.table:
            return self.table[pos_hash].get('best_move')
        return None
    
    def clear(self) -> None:
        """This is the burn it down and try again function"""
        self.table.clear()
        self.hits = 0
        self.misses = 0
    
    def get_hit_rate(self) -> float:
        """This is my tables batting average if you are a baseball fan"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MoveOrderer:
    """This class helps my bot find the best moves without wasting 
    a ton of time on moves that are garbage"""
    
    def __init__(self):
        self.killer_moves = {}
        self.history_heuristic = {}
        self.countermove_table = {}
        
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def order_moves(
        self,
        board: chess.Board,
        ply: int, 
        hash_move: Optional[chess.Move] = None,
        skip_quiets: bool = False
    ) -> List[chess.Move]:
        """This organizes my bots moves from best to worst allowing
        it to see moves that are the strongest first like checkmates and captures"""
        moves = list(board.legal_moves)
        
        filtered_moves = []
        
        for move in moves:
            if board.is_capture(move):
                if not self._is_bad_capture(board, move):
                    filtered_moves.append(move)
            elif not skip_quiets:
                filtered_moves.append(move)
        
        moves = filtered_moves
        killer_moves_at_depth = self.killer_moves.get(ply, []) 
        last_move = board.peek() if board.move_stack else None
        countermove = self.countermove_table.get(last_move.uci() if last_move else None)
        
        move_scores = []
        for move in moves:
            score = self._score_move(board, move, hash_move, killer_moves_at_depth, countermove)
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in move_scores]
    
    def _is_bad_capture(self, board: chess.Board, move: chess.Move) -> bool:
        """This is a defensive function that makes sure the move is 
        not infact as admiral ackbar wisely state "a trap!" but
         it also never filters out checks since that was an problem that
          took 2 days to figure out that it was doing. """
        
        if board.gives_check(move):
            return False
        
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if not victim or not attacker:
            return False
        
        victim_value = self.piece_values[victim.piece_type]
        attacker_value = self.piece_values[attacker.piece_type]
        
        if attacker_value > victim_value:
            board.push(move)
            is_attacked = board.is_attacked_by(board.turn, move.to_square)
            is_check = board.is_check()
            board.pop()
            
            if is_attacked and not is_check:
                return True
        
        return False
    
    def _score_move(
        self,
        board: chess.Board,
        move: chess.Move,
        hash_move: Optional[chess.Move],
        killer_moves: List[chess.Move],
        countermove: Optional[str]
    ) -> int:
        """This function is used to score moves to order them better"""
        
        if hash_move and move == hash_move:
            return 10_000_000
        
        if board.gives_check(move):
            board.push(move)
            is_checkmate = board.is_checkmate()
            board.pop()
            
            if is_checkmate:
                return 5_000_000 
            else:
                return 1_000_000 
        
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                score = 100_000 + (10 * self.piece_values[victim.piece_type] - 
                                   self.piece_values[attacker.piece_type])
                return score
        
        if move.promotion:
            return 90_000 + self.piece_values.get(move.promotion, 0)

        if move in killer_moves:
            return 80_000 if killer_moves and killer_moves[0] == move else 70_000
        
        if countermove and move.uci() == countermove:
            return 60_000
        
        key = (move.from_square, move.to_square)
        score = self.history_heuristic.get(key, 0)
        
        return score
    
    def order_loud_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """I had to research to fully understand quiescence search to get my bot to
        look for only captures, checks, and promotions. This kinda is like a edit of an
        action movie that only has explosions and cool entrances"""
        move_scores = []
        
        for move in moves:
            score = 0
            
            if board.gives_check(move):
                board.push(move)
                is_checkmate = board.is_checkmate()
                board.pop()
                
                if is_checkmate:
                    score = 2_000_000
                else:
                    score = 100_000
            
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += (10 * self.piece_values[victim.piece_type] - 
                            self.piece_values[attacker.piece_type])
            
            if move.promotion:
                score += 800
            
            move_scores.append((score, move))
        
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in move_scores]
    
    def update_cutoff(self, move: chess.Move, ply: int, last_opponent_move: Optional[chess.Move]) -> None:
        """Basically this makes my bot smarter, whenever it hits a 
        move that is super good as in it ends a search early then it remembers it
        for future reference"""
        if ply not in self.killer_moves:
            self.killer_moves[ply] = []
        
        if move not in self.killer_moves[ply]:
            self.killer_moves[ply].insert(0, move)
            self.killer_moves[ply] = self.killer_moves[ply][:2]
        
        key = (move.from_square, move.to_square)
        self.history_heuristic[key] = self.history_heuristic.get(key, 0) + ply * ply
        
        if last_opponent_move:
            self.countermove_table[last_opponent_move.uci()] = move.uci()
    
    def clear(self) -> None:
        self.killer_moves.clear()
        self.history_heuristic.clear()
        self.countermove_table.clear()


class SearchEngine:
    """This is my finalized version of my chess bot's search engine which
    uses a lot of neat little tricks to be both efficent and strong"""
    
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
        self.transposition_table = TranspositionTable(max_size=100000)
        self.move_orderer = MoveOrderer()
        
        self.nodes_searched = 0
        self.cutoffs = 0
        self.lmr_searches = 0
        self.null_move_cutoffs = 0
        
        self.LMR_FULL_DEPTH_MOVES = 4
        self.LMR_REDUCTION_LIMIT = 3
    
    def iterative_deepening_search(self, board: chess.Board, time_limit: float) -> chess.Move:
        """This function implements an iterative deepending search
        with a hard time cut off to serve as a buffer to running out of time.
        It makes sure there is enough time for the next depth level before wasting
        time investigating it. Also it stops the second it finds a checkmate move"""
        
        if board.is_game_over():
            return None
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        start_time = time.time()
        best_move = None
        self.nodes_searched = 0
        self.cutoffs = 0
        self.lmr_searches = 0
        self.null_move_cutoffs = 0
        
        prev_score = 0
        time_per_depth = []
        
        hard_deadline = start_time + (time_limit * 0.85)
        
        for depth in range(1, 30):
            depth_start = time.time()
            
            if time.time() >= hard_deadline:
                print(f" Hit time deadline at depth {depth-1}")
                break
            
            if depth > 3 and time_per_depth:
                avg_time = sum(time_per_depth[-3:]) / len(time_per_depth[-3:])
                predicted_time = avg_time * 3
                
                if time.time() + predicted_time > hard_deadline:
                    print(f" Overshoot likely, stopping at depth {depth-1}")
                    break
            
            if depth >= 4:
                window = 50
                alpha = prev_score - window
                beta = prev_score + window
                
                move, score = self._search_root_aspiration(board, depth, alpha, beta, hard_deadline)
                
                if score <= alpha or score >= beta:
                    if time.time() >= hard_deadline:
                        break
                    move, score = self._search_root(board, depth, hard_deadline)
            else:
                move, score = self._search_root(board, depth, hard_deadline)
            
            depth_time = time.time() - depth_start
            time_per_depth.append(depth_time)
            
            if move:
                best_move = move
                prev_score = score
                print(f"      Depth {depth}: {move} (score: {score:+.0f}, nodes: {self.nodes_searched}, time: {depth_time:.2f}s)")
            
            if abs(score) > 90000:
                print(f"      Found forced mate :D!")
                break
            
            if time.time() >= hard_deadline:
                print(f"      Warning, emergency stop at depth {depth}")
                break
        
        if best_move is None:
            best_move = legal_moves[0]
        
        self.print_search_stats()
        return best_move
    
    def _search_root_aspiration(self, board: chess.Board, depth: int, alpha: float, beta: float, deadline: float) -> Tuple[Optional[chess.Move], float]:
        """Implements a simple root search with an aspiration window"""
        best_move = None
        best_score = float('-inf')
        
        pos_hash = chess.polyglot.zobrist_hash(board)
        hash_move = self.transposition_table.get_best_move(pos_hash)
        ordered_moves = self.move_orderer.order_moves(board, ply=0, hash_move=hash_move)
        
        for move in ordered_moves:
            if time.time() >= deadline:
                break
            
            board.push(move)
            self.nodes_searched += 1
            
            score = -self._pvs(board, depth - 1, -beta, -alpha, search_depth=1, deadline=deadline)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break
        
        return best_move, best_score
    
    def _search_root(self, board: chess.Board, depth: int, deadline: float) -> Tuple[Optional[chess.Move], float]:
        """A simple root search with a full window"""
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        pos_hash = chess.polyglot.zobrist_hash(board)
        hash_move = self.transposition_table.get_best_move(pos_hash)
        ordered_moves = self.move_orderer.order_moves(board, ply=0, hash_move=hash_move)
        
        for i, move in enumerate(ordered_moves):
            if time.time() >= deadline:
                break
            
            board.push(move)
            self.nodes_searched += 1
            
            if i == 0:
                score = -self._pvs(board, depth - 1, -beta, -alpha, search_depth=1, deadline=deadline)
            else:
                score = -self._pvs(board, depth - 1, -alpha - 1, -alpha, search_depth=1, deadline=deadline)
                if score > alpha:
                    score = -self._pvs(board, depth - 1, -beta, -alpha, search_depth=1, deadline=deadline)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
        
        return best_move, best_score
    
    def _pvs(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        search_depth: int,
        deadline: float
    ) -> float:
        """This bad boy is the main thinking function, it:
        - checks time to make sure its not out of it
        - if checkmate is near then it can adjust how it reacts
        - looks at the transportation table to see if it has seen the current positions before
        - Checks for checks to make sure king is safe
        - Razoring which basically is double checking a position because it looks so bad
        - Null pruning checks to see if the current positioning is so good that it will still be strong if 
        the chess bot theoretically passes its turn
        - Futility pruning, basically if a position is super horid then skip it
        - Move count pruning, basically skips looking at moves after a certain point
        - Late Move Reduction, basically a way to look in depth for good moves the 
        first few moves get deeply looked at and then the rest get skimmed until something
        looks interesting and then it goes back and checks them indepth
        - PVS, basically the first move is probably the best so do a full search on it,
        other moves get a quick null move look, if something pops out then go back and search it fully"""
        
        if time.time() >= deadline:
            return self.evaluator.evaluate(board)
        
        mate_value = 100000 - search_depth
        if mate_value < beta:
            beta = mate_value
            if alpha >= mate_value:
                return mate_value
        
        mate_value = -100000 + search_depth
        if mate_value > alpha:
            alpha = mate_value
            if beta <= mate_value:
                return mate_value
        
        pos_hash = chess.polyglot.zobrist_hash(board)
        tt_score = self.transposition_table.probe(pos_hash, depth, alpha, beta)
        if tt_score is not None:
            return tt_score
        
        if board.is_game_over():
            if board.is_checkmate():
                return -100000 + search_depth
            return 0

        in_check = board.is_check()
        if depth == 0:
            if in_check:
                depth = 1
            else:
                return self._quiescence_search(board, alpha, beta, deadline=deadline)
        
        if depth <= 3 and not in_check and abs(alpha) < 90000:
            static_eval = self.evaluator.evaluate(board)
            razor_margin = 300 * depth
            
            if static_eval + razor_margin < alpha:
                q_score = self._quiescence_search(board, alpha, beta, deadline=deadline)
                if q_score < alpha:
                    return q_score
        
        if depth >= 3 and not in_check and search_depth > 1:
            static_eval = self.evaluator.evaluate(board)
            
            reduction = 3
            if static_eval > beta + 200:
                reduction = 4
            
            old_ep = board.ep_square
            board.turn = not board.turn
            board.ep_square = None
            
            null_score = -self._pvs(board, depth - reduction - 1, -beta, -beta + 1, search_depth + 1, deadline)
            
            board.turn = not board.turn
            board.ep_square = old_ep
            
            if null_score >= beta:
                self.null_move_cutoffs += 1
                return beta
        
        if depth <= 2 and not in_check:
            static_eval = self.evaluator.evaluate(board)
            futility_margin = 200 * depth
            
            if static_eval + futility_margin < alpha:
                return alpha
        
        hash_move = self.transposition_table.get_best_move(pos_hash)
        ordered_moves = self.move_orderer.order_moves(board, ply=search_depth, hash_move=hash_move)
        
        max_eval = float('-inf')
        best_move = None
        last_opponent_move = board.peek() if board.move_stack else None
        bound_type = TranspositionTable.UPPER_BOUND
        
        moves_searched = 0
        
        for i, move in enumerate(ordered_moves):
            if time.time() >= deadline:
                break
            
            if (depth <= 3 and not in_check and moves_searched >= 3 + depth * depth 
                and not board.is_capture(move) and not board.gives_check(move)):
                continue
            
            board.push(move)
            self.nodes_searched += 1
            moves_searched += 1
            
            do_full_search = True
            
            if (depth >= 3 and i >= self.LMR_FULL_DEPTH_MOVES 
                and not in_check and not board.is_check()
                and not board.is_capture(move) and not move.promotion):
                
                reduction = 1
                if i >= 6:
                    reduction = 2
                if i >= 12:
                    reduction = min(3, self.LMR_REDUCTION_LIMIT)
                
                self.lmr_searches += 1
                eval_score = -self._pvs(board, depth - reduction - 1, -alpha - 1, -alpha, search_depth + 1, deadline)
                
                do_full_search = (eval_score > alpha)
            
            if i == 0:
                eval_score = -self._pvs(board, depth - 1, -beta, -alpha, search_depth + 1, deadline)
            elif do_full_search:
                eval_score = -self._pvs(board, depth - 1, -alpha - 1, -alpha, search_depth + 1, deadline)
                
                if eval_score > alpha and eval_score < beta:
                    eval_score = -self._pvs(board, depth - 1, -beta, -alpha, search_depth + 1, deadline)
            
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
                if eval_score > alpha:
                    alpha = eval_score
                    bound_type = TranspositionTable.EXACT

                if not board.is_capture(move) and not move.promotion:
                    key = (move.from_square, move.to_square)
                    
                    if eval_score > alpha - 100:
                        bonus = search_depth * search_depth
                    else:
                        bonus = search_depth
                    
                    self.move_orderer.history_heuristic[key] = \
                        self.move_orderer.history_heuristic.get(key, 0) + bonus
            if alpha >= beta:
                self.cutoffs += 1
                bound_type = TranspositionTable.LOWER_BOUND
                self.move_orderer.update_cutoff(move, search_depth, last_opponent_move)
                break
            
        self.transposition_table.store(pos_hash, max_eval, depth, best_move, bound_type)
        
        return max_eval
    
    def _quiescence_search(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        depth: int = 0,
        max_q_depth: int = 10,
        deadline: float = float('inf')
    ) -> float:
        """ Basically this nifty little function keeps the search going AFTER cool move like
        captures, checks, and promotions happen to ensure my bot does not miss tactically
        important moves that can occur after one of these moves. Basically it keeps searching
        until its reaches a unexciting state where nothing big is happening and then finishes.
        it has the following smart features:
        - Stand pat: basically if the position is already good then don't do anything
        - Delta Prune: used to identify really bad positions even when my bot does someting cool
        like capture the queen
        - Futility pruning: kinda like delta pruning but looks to see if my bot is losing badly and if so then stop
        wasting time looking at bad positions.
        - Dynamic depth: Allows my bot to adjust how deep it searches based on the time remaining.
        - SEE filter: makes sure captures that are found are actually worth playing, trading a queen for a pawn is dumb for example.
        """
       
        if time.time() >= deadline:
            return self.evaluator.evaluate(board)
    
        time_remaining = deadline - time.time()
        
        if time_remaining < 0.5:
            max_q_depth = min(max_q_depth, 4)
        elif time_remaining < 1.5:
            max_q_depth = min(max_q_depth, 6)
        elif time_remaining < 5.0:
            max_q_depth = min(max_q_depth, 8)
        
        if depth >= max_q_depth:
            return self.evaluator.evaluate(board)
        
        stand_pat = self.evaluator.evaluate(board)
        
        if stand_pat >= beta:
            return beta
        
        if depth >= 3:
            if stand_pat < -1200:  
                return stand_pat
            elif stand_pat < -800 and depth >= 5: 
                return stand_pat
        
        if depth >= 6 and stand_pat < -600:
            return stand_pat
        
        BIG_DELTA = 900  
        if stand_pat < alpha - BIG_DELTA:
            return alpha
        
        if alpha < stand_pat:
            alpha = stand_pat
        
        if board.is_check():
            loud_moves = list(board.legal_moves)
        else:
            loud_moves = []
            for move in board.legal_moves:
                if board.gives_check(move):
                    loud_moves.append(move)
                elif board.is_capture(move):
                    if not self.move_orderer._is_bad_capture(board, move):
                        loud_moves.append(move)
        
        loud_moves = self.move_orderer.order_loud_moves(board, loud_moves)
        
        for move in loud_moves:
            if time.time() >= deadline:
                break
            
            board.push(move)
            self.nodes_searched += 1
            
            score = -self._quiescence_search(
                board, -beta, -alpha, depth + 1, max_q_depth, deadline
            )
            
            board.pop()
            
            if score >= beta:
                return beta
            
            if score > alpha:
                alpha = score
        
        return alpha
    
    def clear_tables(self) -> None:
        """Used to clear tables"""
        self.transposition_table.clear()
        self.move_orderer.clear()
    
    def get_stats(self) -> dict:
        """I was using this to improve my search agaisnt stockfish"""
        return {
            'nodes_searched': self.nodes_searched,
            'cutoffs': self.cutoffs,
            'null_move_cutoffs': self.null_move_cutoffs,
            'lmr_searches': self.lmr_searches,
            'transposition_hits': self.transposition_table.hits,
            'transposition_misses': self.transposition_table.misses,
            'transposition_hit_rate': self.transposition_table.get_hit_rate(),
            'table_size': len(self.transposition_table.table),
        }
    
    def get_cutoff_rate(self) -> float:
        """helped me get my search cut off rate"""
        if self.nodes_searched == 0:
            return 0.0
        return self.cutoffs / self.nodes_searched

    def print_search_stats(self) -> None:
        """Printed search statistics for debugging"""
        cutoff_rate = self.get_cutoff_rate()
        print(f"\n    Search Stats:")
        print(f"      Nodes searched: {self.nodes_searched:,}")
        print(f"      Cutoffs: {self.cutoffs:,} ({cutoff_rate:.1%})")
        print(f"      Null move cutoffs: {self.null_move_cutoffs:,}")
        print(f"      LMR searches: {self.lmr_searches:,}")
        print(f"      TT hit rate: {self.transposition_table.get_hit_rate():.1%}")