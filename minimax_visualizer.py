"""
I made a separate minimax/alphabeta visualizer to keep my chess bot pure in terms of its purpose.
This file generates a game tree visualization showing minimax values and then the pruned branches.

Its only used for the pre-req portion

Be sure to use the --seed command with a specific value for the same values

- Author: Graeme Huntley
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import argparse
from typing import Tuple, Optional, List, Dict


class GameTreeNode:
    """For nodes in the game tree"""
    def __init__(self, name: str, is_max: bool, value: Optional[int] = None):
        self.name = name
        self.is_max = is_max
        self.value = value
        self.minimax_value = None
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.children = []
        self.pruned = False
        self.on_best_path = False


class GameTree:
    """As the name suggests this builds the game tree"""
    def __init__(self, depth: int = 3, branching_factor: int = 2, 
                 min_val: int = -10, max_val: int = 10):
        self.depth = depth
        self.branching_factor = branching_factor
        self.min_val = min_val
        self.max_val = max_val
        self.node_counter = 0
        self.root = None
    
    def _get_next_name(self) -> str:
        """This makes the node labels"""
        if self.node_counter < 26:
            name = chr(ord('A') + self.node_counter)
        else:
            name = f"N{self.node_counter}"
        self.node_counter += 1
        return name
    
    def build_tree(self) -> GameTreeNode:
        """This constructs a random gametree"""
        self.node_counter = 0
        self.root = self._build_recursive(0, True)
        return self.root
    
    def _build_recursive(self, current_depth: int, is_max: bool) -> GameTreeNode:
        """Recursively constructs the tree"""
        name = self._get_next_name()
        node = GameTreeNode(name, is_max)
        
        if current_depth == self.depth:
            node.value = random.randint(self.min_val, self.max_val)
            node.minimax_value = node.value
        else:

            for _ in range(self.branching_factor):
                child = self._build_recursive(current_depth + 1, not is_max)
                node.children.append(child)
        
        return node


class MinimaxAlgorithm:
    """Implements minimax and AlphaBeta"""
    
    @staticmethod
    def minimax(node: GameTreeNode) -> int:
        """Basic minimax algorithm"""
        if node.value is not None:
            return node.value
        
        if node.is_max:
            max_eval = float('-inf')
            for child in node.children:
                eval_score = MinimaxAlgorithm.minimax(child)
                max_eval = max(max_eval, eval_score)
            node.minimax_value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval_score = MinimaxAlgorithm.minimax(child)
                min_eval = min(min_eval, eval_score)
            node.minimax_value = min_eval
            return min_eval
    
    @staticmethod
    def alpha_beta(node: GameTreeNode, alpha: float = float('-inf'), 
                   beta: float = float('inf')) -> int:
        """Alpha-beta pruning algorithm :D"""
        node.alpha = alpha
        node.beta = beta
        
        if node.value is not None:
            return node.value
        
        if node.is_max:
            max_eval = float('-inf')
            for child in node.children:
                if child.pruned:
                    continue
                    
                eval_score = MinimaxAlgorithm.alpha_beta(child, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    for remaining_child in node.children:
                        if remaining_child not in node.children[:node.children.index(child)+1]:
                            MinimaxAlgorithm._mark_pruned(remaining_child)
                    break
            
            node.minimax_value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                if child.pruned:
                    continue
                    
                eval_score = MinimaxAlgorithm.alpha_beta(child, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    for remaining_child in node.children:
                        if remaining_child not in node.children[:node.children.index(child)+1]:
                            MinimaxAlgorithm._mark_pruned(remaining_child)
                    break
            
            node.minimax_value = min_eval
            return min_eval
    
    @staticmethod
    def _mark_pruned(node: GameTreeNode):
        """Used to mark stuff as pruned"""
        node.pruned = True
        for child in node.children:
            MinimaxAlgorithm._mark_pruned(child)
    
    @staticmethod
    def mark_best_path(node: GameTreeNode):
        """used to show the best path in the tree"""
        node.on_best_path = True
        
        if not node.children:
            return
        
        if node.is_max:
            best_child = max(node.children, key=lambda c: c.minimax_value if c.minimax_value is not None else float('-inf'))
        else:
            best_child = min(node.children, key=lambda c: c.minimax_value if c.minimax_value is not None else float('inf'))
        
        MinimaxAlgorithm.mark_best_path(best_child)


class TreeVisualizer:
    """Visualizes the game tree with NetworkX and matplotlib"""
    
    def __init__(self, root: GameTreeNode, depth: int):
        self.root = root
        self.depth = depth
        self.G = nx.DiGraph()
        self.pos = {}
        self.labels = {}
        self.node_colors = []
        self.edge_colors = []
        self.edge_styles = []
        self.edge_widths = []
        
    def visualize(self, filename: str, show_alphabeta: bool = False, title: str = "Game Tree"):
        """Creates the visualization"""
        self._build_graph()
        self._compute_positions()
        self._prepare_visual_attributes(show_alphabeta)
        self._draw(filename, show_alphabeta, title)
    
    def _build_graph(self):
        """Build NetworkX graph from tree"""
        self._add_nodes_recursive(self.root, 0)
    
    def _add_nodes_recursive(self, node: GameTreeNode, depth: int):
        """Recursively add nodes and edges"""
        self.G.add_node(node.name, node=node, depth=depth)
        
        for child in node.children:
            self.G.add_edge(node.name, child.name)
            self._add_nodes_recursive(child, depth + 1)
    
    def _compute_positions(self):
        """Compute node positions"""
        level_counts = {}
        level_positions = {}
        
        def count_nodes(node: GameTreeNode, depth: int):
            if depth not in level_counts:
                level_counts[depth] = 0
                level_positions[depth] = 0
            level_counts[depth] += 1
            for child in node.children:
                count_nodes(child, depth + 1)
        
        count_nodes(self.root, 0)
        
        def assign_positions(node: GameTreeNode, depth: int, x_start: float, x_end: float):
            x_pos = (x_start + x_end) / 2
            y_pos = -depth
            self.pos[node.name] = (x_pos, y_pos)
            
            if node.children:
                child_width = (x_end - x_start) / len(node.children)
                for i, child in enumerate(node.children):
                    child_x_start = x_start + i * child_width
                    child_x_end = child_x_start + child_width
                    assign_positions(child, depth + 1, child_x_start, child_x_end)
        
        assign_positions(self.root, 0, 0, 10)
    
    def _prepare_visual_attributes(self, show_alphabeta: bool):
        """handles the colors for nodes and edges"""
        for node_name in self.G.nodes():
            node = self.G.nodes[node_name]['node']
            
            # Notice the pretty colors :D
            if node.pruned:
                self.node_colors.append('#FFB6C6')  
            elif node.value is not None:
                self.node_colors.append('#90EE90')  
            else:
                self.node_colors.append('lightblue')
        
        for edge in self.G.edges():
            parent_name, child_name = edge
            parent_node = self.G.nodes[parent_name]['node']
            child_node = self.G.nodes[child_name]['node']
            
            on_best_path = parent_node.on_best_path and child_node.on_best_path
            
            if child_node.pruned and show_alphabeta:
                self.edge_colors.append('red')
                self.edge_styles.append('dashed')
                self.edge_widths.append(2)
            elif on_best_path:
                self.edge_colors.append('green')
                self.edge_styles.append('solid')
                self.edge_widths.append(4)
            else:
                self.edge_colors.append('black')
                self.edge_styles.append('solid')
                self.edge_widths.append(2)
        
        for node_name in self.G.nodes():
            node = self.G.nodes[node_name]['node']
            
            if node.value is not None:
                if node.pruned:
                    self.labels[node_name] = 'X'
                else:
                    self.labels[node_name] = str(node.value)
            else:
                if node.minimax_value is not None:
                    self.labels[node_name] = f"{node.name}\n{node.minimax_value}"
                else:
                    self.labels[node_name] = node.name
    
    def _draw(self, filename: str, show_alphabeta: bool, title: str):
        """Draws the graph if that wasn't clear based on the name"""
        plt.figure(figsize=(16, 10))
        
        for i, (edge, color, style, width) in enumerate(zip(self.G.edges(), 
                                                             self.edge_colors, 
                                                             self.edge_styles, 
                                                             self.edge_widths)):
            nx.draw_networkx_edges(self.G, self.pos, [edge], 
                                  edge_color=color, 
                                  style=style, 
                                  width=width,
                                  arrows=False)
        
        nx.draw_networkx_nodes(self.G, self.pos, 
                              node_color=self.node_colors,
                              node_size=1500,
                              edgecolors='black',
                              linewidths=2)
        
        nx.draw_networkx_labels(self.G, self.pos, self.labels, 
                               font_size=10, font_weight='bold')
        
        for depth in range(self.depth + 1):
            label = "MAX" if depth % 2 == 0 else "MIN"
            plt.text(-1, -depth, label, fontsize=14, fontweight='bold', 
                    ha='right', va='center')
        
        mode = "Alpha-Beta Pruning" if show_alphabeta else "Minimax"
        plt.title(f"{title}\n{mode}", fontsize=16, fontweight='bold')
        
        # Makes the nice legand for easier picture reading :D
        if show_alphabeta:
            legend_elements = [
                mpatches.Patch(color='#90EE90', label='Leaf Node'),
                mpatches.Patch(color='lightblue', label='Internal Node'),
                mpatches.Patch(color='#FFB6C6', label='Pruned Node'),
                plt.Line2D([0], [0], color='green', linewidth=4, label='Best Path'),
                plt.Line2D([0], [0], color='red', linewidth=2, linestyle='dashed', label='Pruned Branch')
            ]
        else:
            legend_elements = [
                mpatches.Patch(color='#90EE90', label='Leaf Node'),
                mpatches.Patch(color='lightblue', label='Internal Node'),
                plt.Line2D([0], [0], color='green', linewidth=4, label='Best Path')
            ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {filename}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Minimax and Alpha-Beta Pruning')
    parser.add_argument('--depth', type=int, default=3, 
                       help='Depth of the game tree (default: 3)')
    parser.add_argument('--branching', type=int, default=2,
                       help='Branching factor (default: 2)')
    parser.add_argument('--min', type=int, default=-5,
                       help='Minimum leaf value (default: -5)')
    parser.add_argument('--max', type=int, default=10,
                       help='Maximum leaf value (default: 10)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='game_tree',
                       help='Output filename prefix (default: game_tree)')
    parser.add_argument('--title', type=str, default='Game Tree',
                       help='Title for the visualization')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎲 Using random seed: {args.seed}")
    
    print(f"\n🌳 Building game tree (depth={args.depth}, branching={args.branching})...")
    tree = GameTree(depth=args.depth, branching_factor=args.branching,
                   min_val=args.min, max_val=args.max)
    root = tree.build_tree()
    
    print("\n📊 Running Minimax algorithm...")
    root_copy1 = tree.build_tree()
    random.seed(args.seed) 
    root_copy1 = tree.build_tree()
    
    minimax_value = MinimaxAlgorithm.minimax(root_copy1)
    MinimaxAlgorithm.mark_best_path(root_copy1)
    print(f"   Minimax value at root: {minimax_value}")
    
    visualizer1 = TreeVisualizer(root_copy1, args.depth)
    visualizer1.visualize(f"{args.output}_minimax.png", 
                         show_alphabeta=False,
                         title=f"{args.title} - Minimax")
    
    print("\n✂️  Running Alpha-Beta Pruning...")
    random.seed(args.seed)
    root_copy2 = tree.build_tree()
    
    ab_value = MinimaxAlgorithm.alpha_beta(root_copy2)
    MinimaxAlgorithm.mark_best_path(root_copy2)
    print(f"   Alpha-Beta value at root: {ab_value}")
    
    visualizer2 = TreeVisualizer(root_copy2, args.depth)
    visualizer2.visualize(f"{args.output}_alphabeta.png", 
                         show_alphabeta=True,
                         title=f"{args.title} - Alpha-Beta Pruning")
    
    print("\n✅ Done! Generated two visualizations:")
    print(f"   - {args.output}_minimax.png")
    print(f"   - {args.output}_alphabeta.png")


if __name__ == "__main__":
    main()