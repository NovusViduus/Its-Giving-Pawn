#!/usr/bin/env python3
"""
visualize_chess_tree.py - Chess Game Tree Visualizer

This creates actual chess game tree visualizations using my real bot components
instead of abstract values. It starts from famous opening positions and shows
how my evaluation function and alpha-beta pruning actually work on real chess positions.

Author: Graeme Huntley
Course: CS5100 - Foundations of AI
"""

import chess
import chess.svg
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse
from typing import List, Optional, Tuple, Dict
from evaluator import Evaluator
from search_engine import MoveOrderer

OPENINGS = {
    'queens_gambit_declined': {
        'name': "Queen's Gambit Declined",
        'moves': ['d2d4', 'd7d5', 'c2c4', 'e7e6'],
        'notation': '1. d4 d5 2. c4 e6'
    },
    'ruy_lopez': {
        'name': 'Ruy Lopez - Morphy Defense',
        'moves': ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6'],
        'notation': '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6'
    },
    'sicilian_four_knights': {
        'name': 'Sicilian Defense - Four Knights',
        'moves': ['e2e4', 'c7c5', 'g1f3', 'e7e6', 'd2d4', 'c5d4', 'f3d4', 'g8f6', 'b1c3', 'b8c6'],
        'notation': '1. e4 c5 2. Nf3 e6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 Nc6'
    },
    'vienna_frankenstein': {
        'name': 'Vienna Game - Frankenstein-Dracula',
        'moves': ['e2e4', 'e7e5', 'b1c3', 'g8f6', 'f1c4', 'f6e4'],
        'notation': '1. e4 e5 2. Nc3 Nf6 3. Bc4 Nxe4'
    },
}


class ChessTreeNode:
    """Node in the actual chess game tree"""
    def __init__(self, board: chess.Board, move_san: str = "START", depth: int = 0, is_max: bool = True):
        self.board = board.copy()
        self.move_san = move_san
        self.depth = depth
        self.is_max = is_max
        self.eval_score = None
        self.minimax_value = None
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.children = []
        self.pruned = False
        self.caused_cutoff = False
        self.on_best_path = False
        
    def __repr__(self):
        return f"Node({self.move_san}, depth={self.depth}, eval={self.eval_score}, minimax={self.minimax_value})"


class ChessTreeBuilder:
    """Builds a real chess game tree using my bot's components"""
    
    def __init__(self, evaluator: Evaluator, move_orderer: MoveOrderer, max_depth: int = 4, top_n_moves: int = 3):
        """
        Initializes the tree builder with my actual bot components.
        
        This uses the real evaluation function and move ordering logic instead of
        random values, so the tree shows how my bot actually thinks about positions.
        """
        self.evaluator = evaluator
        self.move_orderer = move_orderer
        self.max_depth = max_depth
        self.top_n_moves = top_n_moves
        self.node_count = 0
        
    def build_tree(self, root_board: chess.Board) -> ChessTreeNode:
        """
        Builds a game tree starting from the given position.
        
        Uses my move orderer to pick the top N moves at each node, which means
        the tree shows the moves my bot would actually consider instead of random garbage.
        """
        self.node_count = 0
        root = ChessTreeNode(root_board, "START", depth=0, is_max=True)
        self._build_recursive(root)
        return root
    
    def _build_recursive(self, node: ChessTreeNode):
        """
        Recursively builds the tree using my move ordering to pick top moves.
        
        At leaf nodes, it evaluates the position with my actual evaluation function
        so the scores reflect what my bot actually thinks about those positions.
        """
        if node.depth >= self.max_depth:
            node.eval_score = self.evaluator.evaluate(node.board)
            node.minimax_value = node.eval_score
            return
        
        ordered_moves = self.move_orderer.order_moves(
            node.board, 
            ply=node.depth,
            hash_move=None,
            skip_quiets=False
        )
        
        top_moves = ordered_moves[:self.top_n_moves]
        
        if not top_moves:
            if node.board.is_checkmate():
                node.eval_score = -10000 if node.is_max else 10000
            else:
                node.eval_score = 0
            node.minimax_value = node.eval_score
            return
        
        for move in top_moves:
            move_san = node.board.san(move)
            node.board.push(move)
            child = ChessTreeNode(
                node.board, 
                move_san, 
                depth=node.depth + 1,
                is_max=not node.is_max
            )
            node.children.append(child)
            
            node.board.pop()
            
            self._build_recursive(child)
            
            self.node_count += 1


class MinimaxSolver:
    """Solves the tree with Minimax and Alpha-Beta pruning"""
    
    @staticmethod
    def minimax(node: ChessTreeNode) -> int:
        """
        Basic minimax without pruning - just for comparison.
        
        This is the baseline algorithm that searches everything before we add
        the alpha-beta optimizations that make it way faster.
        """
        if node.eval_score is not None:
            return node.eval_score
        
        if node.is_max:
            max_eval = float('-inf')
            for child in node.children:
                eval_score = MinimaxSolver.minimax(child)
                max_eval = max(max_eval, eval_score)
            node.minimax_value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval_score = MinimaxSolver.minimax(child)
                min_eval = min(min_eval, eval_score)
            node.minimax_value = min_eval
            return min_eval
    
    @staticmethod
    def alpha_beta(node: ChessTreeNode, alpha: float = float('-inf'), beta: float = float('inf')) -> int:
        """
        Alpha-beta pruning algorithm that cuts off useless branches.
        
        This is the smart version that realizes when a branch can't possibly
        affect the final decision and skips searching it. Way more efficient
        than plain minimax, especially on deep trees.
        """
        node.alpha = alpha
        node.beta = beta
        
        if node.eval_score is not None:
            return node.eval_score
        
        if node.is_max:
            max_eval = float('-inf')
            for i, child in enumerate(node.children):
                if child.pruned:
                    continue
                
                eval_score = MinimaxSolver.alpha_beta(child, alpha, beta)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    child.caused_cutoff = True
                    for remaining_child in node.children[i+1:]:
                        MinimaxSolver._mark_pruned(remaining_child)
                    break
            
            node.minimax_value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for i, child in enumerate(node.children):
                if child.pruned:
                    continue
                
                eval_score = MinimaxSolver.alpha_beta(child, alpha, beta)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    child.caused_cutoff = True
                    for remaining_child in node.children[i+1:]:
                        MinimaxSolver._mark_pruned(remaining_child)
                    break
            
            node.minimax_value = min_eval
            return min_eval
    
    @staticmethod
    def _mark_pruned(node: ChessTreeNode):
        """Marks a node and its entire subtree as pruned"""
        node.pruned = True
        for child in node.children:
            MinimaxSolver._mark_pruned(child)
    
    @staticmethod
    def mark_best_path(node: ChessTreeNode):
        """Marks the principal variation (best path) through the tree"""
        node.on_best_path = True
        
        if not node.children or all(c.pruned for c in node.children):
            return
        
        valid_children = [c for c in node.children if not c.pruned]
        if not valid_children:
            return
        
        if node.is_max:
            best_child = max(valid_children, key=lambda c: c.minimax_value if c.minimax_value is not None else float('-inf'))
        else:
            best_child = min(valid_children, key=lambda c: c.minimax_value if c.minimax_value is not None else float('inf'))
        
        MinimaxSolver.mark_best_path(best_child)


class ChessTreeVisualizer:
    """Visualizes the chess game tree with NetworkX and matplotlib"""
    
    def __init__(self, root: ChessTreeNode, opening_name: str):
        """
        Sets up the visualizer for a chess game tree.
        
        This creates the graph structure and prepares all the visual attributes
        like colors, edge styles, and labels.
        """
        self.root = root
        self.opening_name = opening_name
        self.G = nx.DiGraph()
        self.pos = {}
        self.labels = {}
        self.node_colors = []
        self.edge_colors = []
        self.edge_styles = []
        self.edge_widths = []
        self.edge_labels = {}
        
    def visualize(self, filename_base: str):
        """
        Creates both minimax and alpha-beta visualizations.
        
        Generates two separate images - one showing plain minimax values and one
        showing the alpha-beta pruning with cut branches marked in red.
        """
        self._build_graph()
        self._compute_positions()
        
        self._prepare_minimax_visual()
        self._draw(f"{filename_base}_minimax.png", show_pruning=False, 
                  title=f"{self.opening_name} - Minimax Search")
        
        self._prepare_alphabeta_visual()
        self._draw(f"{filename_base}_alphabeta.png", show_pruning=True,
                  title=f"{self.opening_name} - Alpha-Beta Pruning")
    
    def _build_graph(self):
        """Builds the NetworkX graph from the tree structure"""
        node_id = [0]
        self._add_nodes_recursive(self.root, node_id)
    
    def _add_nodes_recursive(self, node: ChessTreeNode, node_id: List[int]):
        """Recursively adds nodes and edges to the graph"""
        current_id = node_id[0]
        node_id[0] += 1
        
        self.G.add_node(current_id, node=node)
        
        for child in node.children:
            child_id = node_id[0]
            self.G.add_edge(current_id, child_id, move=child.move_san)
            self._add_nodes_recursive(child, node_id)
    
    def _compute_positions(self):
        """
        Computes node positions for a clean tree layout.
        
        Uses a hierarchical layout where each level is evenly spaced and nodes
        are positioned to avoid overlaps.
        """
        def assign_positions(node_id: int, depth: int, x_start: float, x_end: float):
            node = self.G.nodes[node_id]['node']
            
            x_pos = (x_start + x_end) / 2
            y_pos = -depth * 2
            self.pos[node_id] = (x_pos, y_pos)
            
            children = list(self.G.successors(node_id))
            if children:
                child_width = (x_end - x_start) / len(children)
                for i, child_id in enumerate(children):
                    child_x_start = x_start + i * child_width
                    child_x_end = child_x_start + child_width
                    assign_positions(child_id, depth + 1, child_x_start, child_x_end)
        
        max_width = 3 ** self.root.depth
        assign_positions(0, 0, 0, max_width * 2)
    
    def _prepare_minimax_visual(self):
        """Prepares visual attributes for minimax visualization (no pruning shown)"""
        self.node_colors = []
        self.edge_colors = []
        self.edge_styles = []
        self.edge_widths = []
        self.labels = {}
        self.edge_labels = {}
        
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]['node']
            
            if node.eval_score is not None:
                self.node_colors.append('#90EE90')
            else:
                self.node_colors.append('lightblue')
            
            if node.eval_score is not None:
                self.labels[node_id] = f"{node.move_san}\n{node.eval_score:+d}"
            else:
                if node.minimax_value is not None:
                    self.labels[node_id] = f"{node.move_san}\n{node.minimax_value:+d}"
                else:
                    self.labels[node_id] = node.move_san
        
        for edge in self.G.edges():
            parent_id, child_id = edge
            child_node = self.G.nodes[child_id]['node']
            parent_node = self.G.nodes[parent_id]['node']
            
            on_best_path = parent_node.on_best_path and child_node.on_best_path
            
            if on_best_path:
                self.edge_colors.append('green')
                self.edge_styles.append('solid')
                self.edge_widths.append(4)
            else:
                self.edge_colors.append('black')
                self.edge_styles.append('solid')
                self.edge_widths.append(2)
            
            move_san = self.G.edges[edge]['move']
            self.edge_labels[edge] = move_san
    
    def _prepare_alphabeta_visual(self):
        """Prepares visual attributes for alpha-beta visualization (with pruning)"""
        self.node_colors = []
        self.edge_colors = []
        self.edge_styles = []
        self.edge_widths = []
        self.labels = {}
        self.edge_labels = {}
        
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]['node']
            
            if node.pruned:
                self.node_colors.append('#FFB6C6')
            elif node.eval_score is not None:
                self.node_colors.append('#90EE90')
            else:
                self.node_colors.append('lightblue')
            
            label = node.move_san
            if node.pruned:
                label += "\nPRUNED"
            elif node.eval_score is not None:
                label += f"\n{node.eval_score:+d}"
            elif node.minimax_value is not None:
                label += f"\n{node.minimax_value:+d}"
            
            self.labels[node_id] = label
        
        for edge in self.G.edges():
            parent_id, child_id = edge
            child_node = self.G.nodes[child_id]['node']
            parent_node = self.G.nodes[parent_id]['node']
            
            if child_node.pruned:
                self.edge_colors.append('red')
                self.edge_styles.append('dashed')
                self.edge_widths.append(2)
            elif parent_node.on_best_path and child_node.on_best_path:
                self.edge_colors.append('green')
                self.edge_styles.append('solid')
                self.edge_widths.append(4)
            else:
                self.edge_colors.append('black')
                self.edge_styles.append('solid')
                self.edge_widths.append(2)
            
            move_san = self.G.edges[edge]['move']
            self.edge_labels[edge] = move_san
    
    def _draw(self, filename: str, show_pruning: bool, title: str):
        """Draws the actual graph and saves it to a file"""
        plt.figure(figsize=(20, 12))
        
        for i, (edge, color, style, width) in enumerate(zip(
            self.G.edges(), self.edge_colors, self.edge_styles, self.edge_widths
        )):
            nx.draw_networkx_edges(
                self.G, self.pos, [edge],
                edge_color=color,
                style=style,
                width=width,
                arrows=True,
                arrowsize=20,
                connectionstyle="arc3,rad=0.1"
            )
        
        nx.draw_networkx_nodes(
            self.G, self.pos,
            node_color=self.node_colors,
            node_size=2000,
            edgecolors='black',
            linewidths=2
        )
        
        nx.draw_networkx_labels(
            self.G, self.pos, self.labels,
            font_size=8, font_weight='bold'
        )
        
        nx.draw_networkx_edge_labels(
            self.G, self.pos, self.edge_labels,
            font_size=7, font_color='darkred'
        )
        
        depth_labels = {}
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]['node']
            depth = node.depth
            if depth not in depth_labels:
                label = "MAX" if node.is_max else "MIN"
                y_pos = -depth * 2
                plt.text(-2, y_pos, label, fontsize=14, fontweight='bold',
                        ha='right', va='center', bbox=dict(boxstyle='round', facecolor='wheat'))
                depth_labels[depth] = True
        
        best_child = None
        for child in self.root.children:
            if child.on_best_path:
                best_child = child
                break
        
        if best_child:
            title += f"\nBest Move: {best_child.move_san} (Score: {self.root.minimax_value:+d} centipawns)"
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        if show_pruning:
            legend_elements = [
                mpatches.Patch(color='#90EE90', label='Leaf Node (Evaluated)'),
                mpatches.Patch(color='lightblue', label='Internal Node'),
                mpatches.Patch(color='#FFB6C6', label='Pruned Node'),
                plt.Line2D([0], [0], color='green', linewidth=4, label='Best Path (PV)'),
                plt.Line2D([0], [0], color='red', linewidth=2, linestyle='dashed', label='Pruned Branch')
            ]
        else:
            legend_elements = [
                mpatches.Patch(color='#90EE90', label='Leaf Node (Evaluated)'),
                mpatches.Patch(color='lightblue', label='Internal Node'),
                plt.Line2D([0], [0], color='green', linewidth=4, label='Best Path (PV)')
            ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")
        plt.close()


def main():
    """Main function that creates the chess tree visualization"""
    parser = argparse.ArgumentParser(
        description='Visualize chess game tree with minimax and alpha-beta pruning'
    )
    parser.add_argument(
        '--opening',
        type=str,
        default='queens_gambit_declined',
        choices=list(OPENINGS.keys()),
        help='Opening position to analyze'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=4,
        help='Search depth (number of half-moves, default: 4)'
    )
    parser.add_argument(
        '--moves',
        type=int,
        default=3,
        help='Number of top moves to consider per node (default: 3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='chess_tree',
        help='Output filename prefix (default: chess_tree)'
    )
    
    args = parser.parse_args()
    
    opening_info = OPENINGS[args.opening]
    
    print("=" * 70)
    print("CHESS GAME TREE VISUALIZATION")
    print("=" * 70)
    print(f"\nOpening: {opening_info['name']}")
    print(f"Moves: {opening_info['notation']}")
    print(f"Search Depth: {args.depth} half-moves")
    print(f"Top Moves per Node: {args.moves}")
    print()
    
    board = chess.Board()
    for move_uci in opening_info['moves']:
        board.push_uci(move_uci)
    
    print("Starting Position:")
    print(board)
    print()
    
    print("Initializing bot components...")
    evaluator = Evaluator()
    move_orderer = MoveOrderer()
    
    print(f"\nBuilding game tree (depth={args.depth}, top {args.moves} moves per node)...")
    builder = ChessTreeBuilder(evaluator, move_orderer, max_depth=args.depth, top_n_moves=args.moves)
    root = builder.build_tree(board)
    print(f"   Created {builder.node_count} nodes")
    
    print("\nRunning Minimax algorithm...")
    minimax_value = MinimaxSolver.minimax(root)
    MinimaxSolver.mark_best_path(root)
    print(f"   Minimax value: {minimax_value:+d} centipawns")
    
    best_child = None
    for child in root.children:
        if child.on_best_path:
            best_child = child
            break
    
    if best_child:
        print(f"   Best move: {best_child.move_san}")
    
    print("\nRunning Alpha-Beta Pruning...")
    root_ab = builder.build_tree(board)
    ab_value = MinimaxSolver.alpha_beta(root_ab)
    MinimaxSolver.mark_best_path(root_ab)
    print(f"   Alpha-Beta value: {ab_value:+d} centipawns")
    
    def count_pruned(node):
        count = 1 if node.pruned else 0
        for child in node.children:
            count += count_pruned(child)
        return count
    
    pruned_count = count_pruned(root_ab)
    total_nodes = builder.node_count
    print(f"   Pruned {pruned_count}/{total_nodes} nodes ({pruned_count/total_nodes*100:.1f}%)")
    
    print("\nCreating visualizations...")
    visualizer = ChessTreeVisualizer(root_ab, opening_info['name'])
    visualizer.visualize(args.output)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - {args.output}_minimax.png (plain minimax search)")
    print(f"  - {args.output}_alphabeta.png (with alpha-beta pruning)")
    print(f"\nNow draw the alpha/beta values on the alpha-beta image!")
    print(f"For each node, show:")
    print(f"  - alpha value on the left")
    print(f"  - beta value on the right")
    print(f"  - Mark where cutoffs occurred")
    print("=" * 70)


if __name__ == "__main__":
    main()