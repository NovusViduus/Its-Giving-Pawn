[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=20889222)
# Homework - Adversarial Search ♔♕♗♘♙♖

Topics: Minimax and AlphaBeta

For this assignment you will be making your own chessbot taking advantage of the search techniques discussed in class. You do not need to program the rules of chess in order to complete this assignment.

---

# First Half - Programming Minimax & AlphaBeta


You must complete this section before moving on to the second half of the homework. You can use GenerativeAI to assist you. Perform the following steps.

1. Create the following search tree and visualize it using NetworkX (credit geeksforgeeks). It is recommended that you label the edges to make the choices clear (like `L` and `R`).

![Game Tree](MIN_MAX1.jpg)

2. Add your [NetworkX](https://networkx.org/) rendering here (it does not need to look exactly like the above image):

![Basic Minimax Tree](game_tree_minimax.png)

3. Create a Minimax tree search function that will accept the tree you generated from Step 1 and compute the minimax value of the tree, updating the value of each node as it goes.

4. Render your new version of the search tree to show the choice ultimately decided upon at the root (i.e. color the left or right edge down to the leaf) after running the minimax algorithm.

5. Create an AlphaBeta tree search function that will accept the tree you generated from 1 and compute the minimax value of the tree using alpha-beta pruning, eliminating edges as it goes.

6. Render your new version of the search tree showing the pruned edges from alpha-beta pruning (i.e. if your algorithm decides not to search an edge color it something different). Add your rendering here:

![AlphaBeta Pruned Tree](game_tree_alphabeta.png)

If you need an example of what it can look like, see the below reference (without the nodes being labeled as A, B, ...) again originally from geeksforgeeks.

![Alpha Beta Pruning](ALPHA_BETA1.jpg)


---

# Second Half


## Part 0 - Pre-req

There are some libraries and other software that you will need.

### Needed Python Packages

* chess - [pypi.org/project/chess/](https://pypi.org/project/chess/) used for modeling boards, identifying legal moves, and faciliating communication. Install with the command `pip install chess`
* pyinstaller - [pyinstaller.org/](https://pyinstaller.org/) for converting your .py files into .exe executables. Install with the command `pip install pyinstaller`
* chester - [pypi.org/project/chester/](https://pypi.org/project/chester/) runs tournaments of chessbots installed with `pip install chester`

```bash
pip install chess pyinstaller chester
```

### Visualizing Games

You can use any visualizer you like to play against an engine. The one we'll recommend is Python Easy Chess GUI (see instructions below) which requires some additional setup.

* PySimpleGUI [github.com/PySimpleGUI/PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI) creates a generic GUI. Install with `pip install pysimplegui`
* Pyperclip [github.com/asweigart/pyperclip](https://github.com/asweigart/pyperclip) allows for copy/paste functionality with the GUI. Install with `pip install pyperclip`
* Python Easy Chess GUI [https://github.com/fsmosca/Python-Easy-Chess-GUI](https://github.com/fsmosca/Python-Easy-Chess-GUI) clone this repository in a second directory and run the command `python python_easy_chess_gui.py` to run the program.

When setup correctly, it will look like:

![example gui visual](python_easy_chess_gui.png)

```bash
pip install pysimplegui pyperclip
python python_easy_chess_gui.py
```

It is a good idea to turn off Book Moves (Book > Set Book > Uncheck "Use book") and to limit the depth of the chessbots (Engine > Set Depth > 12) so that some bots don't spend all their time thinking. To add a chessbot, go to Engine > Manage > Install > Add > then select your .exe executable. Simply select the opponent by going to Engine > Set Engine Opponent > and select your bot. When ready to play, click Mode > Play. Visit [https://lczero.org/play/quickstart/](https://lczero.org/play/quickstart/) for other visualizers.

### Engines and Tournaments

To create your executable agent use the command `pyinstaller --onefile random_chess_bot.py` except replace with your agent file. This will create an executable, like `random_chess_bot.exe`, inside of a new directory called `dist`. For simplicity, move this file to the directory with the tournament code. **If you are on Mac**, there is another way to make this program executable by using `chmod +x random_chess_bot.py` in the terminal, but pyinstaller should work as well.

In order to test your agent, you'll need to run it against at least one other strong chessbot executable. Good candidates include:

* Stockfish - recommended and probably the strongest open source chessbot [https://stockfishchess.org/](https://stockfishchess.org/) **if you are on mac** you can install using the command `brew install stockfish` and then you should be able to simply run the command `stockfish` to start the bot.
* Goldfish - [https://github.com/bsamseth/Goldfish](https://github.com/bsamseth/Goldfish)
* Leela Chess Zero (Lc0) - [https://lczero.org/](https://lczero.org/)

We recommend downloading the executable to the same directory as the chester tournament code. Edit the [tournament.py](tournament.py) file to add your chessbot as a player. You can then run a tournament with `python tournament.py` and wait for the results.

### How Chess Package Works

If you run the following Python code you'll see the output below it.

```python
import chess

board = chess.Board()
print(board)
```

```text
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
```

This is an 8x8 chessboard with the capital letters representing White and lower case for Black. The letter 'P' is for Pawn, 'R' for Rook, 'N' for Knight (not 'K'), 'Q' for Queen, and 'K' for King. The columns are represented with the letters 'a', 'b', 'c', ..., 'h' and rows with the numbers 1 through 8. This means that to give the move Knight on b1 to the spot c3, it is given with the notation Nc3.

The library is able to determine what are the possible valid legal moves allowed by the game with the command `board.legal_moves` which at the start gives:

```text
<LegalMoveGenerator at 0x2283a4b3e80 (Nh3, Nf3, Nc3, Na3, h3, g3, f3, e3, d3, c3, b3, a3, h4, g4, f4, e4, d4, c4, b4, a4)>
```

You can find lots of documentation about all of the functions built into the Python chess library [https://python-chess.readthedocs.io/en/latest/core.html](https://python-chess.readthedocs.io/en/latest/core.html).

If during the development process you wish to visualize the board state like a more traditional image, take a look at `chess.svg` rendering [https://python-chess.readthedocs.io/en/latest/svg.html](https://python-chess.readthedocs.io/en/latest/svg.html). For example, the following code creates the resulting svg graphic.

```python
import chess
import chess.svg

b = chess.Board()
svg = chess.svg.board(b)
f = open("board.svg", "w")
f.write(svg)
f.close()
```

![chess board](board.svg)

You can add images to NetworkX graphs if you like, see [networkx.org/documentation/](https://networkx.org/documentation/stable/auto_examples/drawing/plot_custom_node_icons.html) for more info.

## Part 1 - Instructions

This assignment is meant to ensure that you:

* Understand the concepts of adversarial search
* Can program an agent to traverse a graph along edges
* Experience developing different pruning algorithms
* Apply the basics of Game Theory
* Can argue for chosing one algorithm over another in different contexts

You are tasked with:

0. Copy [random_chess_bot.py](random_chess_bot.py) and update it to develop a new brand new and intelligent chessbot with a unique & non-boring name. ***Do not name it `my_chess_bot`, your name, or something similar.*** If you do, you will ***automatically earn a zero*** for this assignment. Come up with something creative, humourous, witty, adventuous, -- or something will strike fear into the hearts of the other chessbots in this competition.
1. Develop a strong evaluation function for a board state. Take a look at "Programming a Computer for Playing Chess" by Claude Shannon [https://www.computerhistory.org/chess/doc-431614f453dde/](https://www.computerhistory.org/chess/doc-431614f453dde/) published in 1950. You will specifically want to take a look at section 3 in which Shannon describes a straight-forward evaluation function that you can simplify to only evaluate material (pieces) to score a board state.

  * **Note** that your evaluation function will play a crutial role in the strength of your chessbot. It is ok to start with a simple function to get going, but you will need to find ways to improve it because your bot will be competing with the bots from the rest of the class and points are on the line.
  * Talk the teaching team for helpful tips if you are really stuck.
  
2. Alter your chessbot so that when called with the command line parameter `draw` (such as `python random_chess_bot.py draw`) it creates a Minimax visualization that:

* Starts with the root as the end of a named opening sequence such as the Queen's Gambit Declined 1. d4 d5 2. c4 e6 [https://en.wikipedia.org/wiki/Queen%27s_Gambit_Declined](https://en.wikipedia.org/wiki/Queen%27s_Gambit_Declined). This is because in order for a simple evaluation function to have any chance, there needs to be the potential for pieces to be captured. If you don't like the QGD, we can suggest the:
  * [Ruy Lopez - Morphy Defence](https://en.wikipedia.org/wiki/Ruy_Lopez) 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
  * [Four Nights Sicilian Defence](https://www.chess.com/openings/Sicilian-Defense-Four-Knights-Variation) 1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 Nc6
  * [Vienna Game Frankenstein–Dracula Variation](https://en.wikipedia.org/wiki/Vienna_Game,_Frankenstein%E2%80%93Dracula_Variation) 1. e4 e5 2. Nc3 Nf6 3. Bc4 Nxe4
  * Any other opening you like that ends with Black making a move so that it is White's turn.
* Have your graph select the top three moves per node and label each edge with the move's notation.
* Limit the depth of the generated tree visuals to four (4) half-moves ahead (W-B-W-B). This is because the visuals will be too difficult to read otherwise.
* Label the leaf nodes with the result of that board state's evaluation
* Perform the Minimax algorithm on the tree, labeling each node backpropogating with the correct minimax value.
* Identify the final value of the game tree and the move that your bot will select in a title or subtitle.
* Perform Alpha-Beta pruning on this game tree to re-color edges and subtrees that have been pruned.
* Finally, draw on the image (use a tablet or print and mark on it) with the results of alpha and beta for each node -- clearly identifying the why & how your graph pruned these edges that it pruned.
* If no branches were pruned, change your opening and/or your evaluation function so that there is some demonstrable pruning.

3. At any given point in a chess game there are roughly 20 possible moves. Your Minimax and Alpha-Beta Pruning algorithms will spend a lot of time on what are clearly poor moves. You are allowed alter these algorithms slightly to not even consider poor quality moves or to only look at the top 7 to 10 moves at a time.
4. When you are done, answer the questions in the reflection and complete the last two sections.

### Documentation

Ensure that your chessbot follows normal PyDoc specs for documentation and readability.

## Part 2 - Reflection

Update the README to answer the following questions:

1. Describe your experiences implementing these algorithms. What things did you learn? What aspects were a challenge?

Answer:
So I went way overboard with my bot because I was like well if we play stockfish I want to be competative. I ended up spending most of my free time since we were given this assignment just trying to build a bot that could even hold its own against level zero of stockfish. I found the algos to be interesting especially in terms of how you go about increasing speed and depth of searches while preserving the searches effectiveness to be super interesting.

2. These algorithms assumed that you could reach the leaves of the tree and then reverse your way through it to "solve" the game. In our game (chess) that was not feasible. How effective do you feel that the depth limited search with an evaluation function was in selecting good moves? If you play chess, were you able to beat your bot? If so, why did you beat it? If not, what made the bot so strong - the function or the search?

Answer:
I honestly think my evaluation functions are what would have wiped the floor with me. While my search is propably one of the stronger ones in class it is my evaluation functions that add hundreds of elo points to my bot's capabilities. I had to do a ton of research about how specific chess pieces are viewed by top level players and then how to implement these strategies into code. The search allows it to think further ahead in less time than I ever possibly could but without the evaluation functions it would have a harder time beating someone.

3. Shannon wrote "... it is possible for the machine to play legal chess, merely making a randomly chosen legal move at each turn to move. The level of play with such a strategy is unbelievably bad. The writer played a few games against this random strategy and was able to checkmate generally in four or five moves (by fool's mate, etc.)" Did you try playing the provided random chessbot and if so, what this your experience? How did your chessbot do against the random bot in your tests?

Answer;
I as a human never directly played against the random bot but my chessbot demolished both the random bot and the mate_in_one bot in two sets of 200 games it never lost to either one of them. The random bot is by definition astoundingly bad at chess. Sure the moves are legal but chess is all about smart strategic plays not about moving them however you feel like.

4. Explain the what would happen against an opponent who tries to maximize their own utility instead of minimizing yours.

Answer:
I think this mindset would not stop my bot from being able to win as it is designed to counter you no matter how you play. So an opponent that only focused on trying to make the best move for itself possible would miss opprotunities to strike at my bot's positions and would also leave itself open to be picked appart as my bot constantly attacked.

5. What is the "horizon" and how is it used in adversarial tree search?

Answer: The horizon is basically a situation where the agent will push a bad outcome out past its search depth in order to avoid it, however the problem is similar to a child hiding a broken vase from their parents. Just becaue you can't see the problem anymore doesnt mean it is resolved or gone, you just delayed having to face it.

6. (Optional - Not Graded) What did you think of this homework? Challenging? Difficult? Fun? Worth-while? Useful? Etc.?

Answer:
I really loved this assignment albeit that I spent WAY too much time on it because I was trying to fight a david vs goliath level battle. I really liked having to think about how to improve my evaluaton file to make it stronger and how to speed up my search and manage the game time.

---

![vienna_frankenstein MiniMax](chess_tree_minimax.png)
![vienna_frankenstein AlphaBeta](chess_tree_alphabeta.png)

Add the images that you created from the forced opening that you chose so that it demonstrates AlphaBeta Pruning.

Alpha-Beta Pruning Explanation
Alpha-beta pruning works by maintaining two values as we search the tree: alpha (the best score MAX can guarantee) and beta (the best score MIN can guarantee). When beta ≤ alpha, we know the current branch can't affect the final decision, so we cut it off and skip searching the rest.

How it worked on my vienna_frankenstein tree:
At the root (START, MAX node):
Alpha starts at -∞, Beta at +∞
basically we start searching Bxf7's children and we find that the max layer can force -367 for Kxf7 and -26 for Ke7. But this gets countered at the last min level by it choosing Bxf7 with a value of -367. The otherside gets a whole subtree pruned as at the second max level we end up with -88 for Rg8 and Be7 -75. This results in Nd6+ and all of its children getting pruned. The end result of all of this is the root max node picking -88 as the best outcome.

Evaluation function:

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

Please refer to the evaluator.py file to see my full evaluation class, we are talking 2000ish lines of code and that is way too much to paste here.

Conciesly and effictively describe the evaluation function that you used for your chessbot. You can also use Latex as long as you explain the symbols and justify why you created your function in the manner with which you did.

My chess bot uses a sophisticated 12-component evaluation function designed to understand positions the way strong players do, not just count material like a basic engine. The core philosophy is that evaluation needs to be phase-aware - what matters in the opening is totally different from what matters in the endgame.

The 12 Components:

Material + Piece-Square Tables - Foundation of any chess engine. Not just raw piece values, but also WHERE pieces are positioned. A centralized knight is way more valuable than one on the rim.

Piece Safety - Uses Static Exchange Evaluation (SEE) to detect hanging pieces and bad trades. If a piece is attacked and the exchange sequence loses material, that's penalized.

King Safety/Activity - This is phase-dependent, which is critical. In the middlegame, I evaluate king safety: castling status, pawn shield integrity, enemy piece proximity, escape squares. In the endgame, I flip to king activity: centralization, supporting passed pawns, attacking enemy pawns.

Mobility - Quality-weighted mobility, not just move count. Moves that attack enemy pieces or control central squares are weighted higher than random shuffling.

Passed Pawns - Exponentially scaled by advancement rank because a pawn on the 6th rank is WAY more dangerous than one on the 3rd. Also considers king distances, blockers, and rook support (Tarrasch principle).

Bishop Pair - Context-aware bonus. The bishop pair is strong, but only in open positions with active bishops. I check board openness, bishop mobility, and game phase before applying the bonus.

Development - Only matters in the opening. Castling priority, getting minor pieces off the back rank, avoiding premature queen development.

Center Control - Attack count on d4/d5/e4/e5 weighted by piece type. Pawn control is most stable so it counts more than piece control.

Rook Placement - Open files, semi-open files, 7th rank presence, doubled rooks, and rook-queen batteries.

Pawn Structure - Penalizes doubled pawns, isolated pawns, backward pawns, and disconnected pawn islands. Rewards pawn chains and phalanxes.

Knight Outposts - Advanced squares protected by pawns that can't be challenged by enemy pawns. Knights on outposts dominate games.

Connected Rooks - Bonus when rooks defend each other on the same rank/file.

Why This Design:
I built it this way because every component is independently tunable for machine learning optimization. The 12 weights can be trained against Stockfish to learn optimal values instead of hand-tuning forever. Each component captures a distinct positional concept from chess theory, and the phase-awareness ensures the bot knows what actually matters at each stage of the game. The whole system is backed by proper piece-square tables and uses caching for speed since the same positions get evaluated repeatedly due to transpositions.
