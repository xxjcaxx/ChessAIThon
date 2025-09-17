import chess

import math
import random
from chessmodel import predict_chess_move, init_model
import json
import sys

sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr


model, device = init_model()

# Define the MCTS Node class
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Current board state
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this state
        self.children = []  # Child nodes (future possible states)
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Total reward (win/loss/draw) from simulations

    def is_fully_expanded(self):
        # Returns True if all possible moves have been explored
        return len(self.children) == sum(1 for _ in self.state.legal_moves)


    def best_child(self, exploration_weight=1.4):
        # Select the child with the best value using UCT (Upper Confidence Bound for Trees)
        best_value = -float('inf')
        best_node = None
        for child in self.children:
            uct_value = child.value / (child.visits + 1) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node
        
    def to_dict(self, depth=3):
        node_dict = {
            "move": str(self.move),
            "value": self.value,
            "visits": self.visits,
        }
        if depth > 0 and self.children:
            node_dict["children"] = [child.to_dict(depth - 1) for child in self.children]
        return node_dict

    def to_json(self, depth=3):
        """Retorna el nodo y sus hijos (hasta profundidad depth) en JSON."""
        return json.dumps(self.to_dict(depth), indent=2)


# Define the MCTS algorithm
class MCTS:
    def __init__(self, root, get_best_function, simulations=100):
        self.root = root  # Root node
        self.get_best_function = get_best_function  # Function to get the best move
        self.simulations = simulations  # Number of MCTS simulations

    def search(self):
        for _ in range(self.simulations):
            print(_)
            # Step 1: Selection
            node = self._select(self.root)

            # Step 2: Expansion
            if not node.is_fully_expanded():
                node = self._expand(node)

            # Step 3: Simulation (Playout)
            winner = self._simulate(node)

            # Step 4: Backpropagation
            self._backpropagate(node, winner)

        # Return the best move after simulations
        print("Children:", len(self.root.children[0].children))
        print(self.root.to_json())
        return self.root.best_child(exploration_weight=0).move

    def _select(self, node):
        # Traverse down the tree to find a leaf node
        while node.is_fully_expanded():
            node = node.best_child()  # Use the best child with UCT
        return node

    def _expand(self, node):
        # Expand one of the children (possible moves from the current position)
        legal_moves = node.state.legal_moves
        for move in legal_moves:
            new_state = node.state.copy()  # Clone the board to simulate the move
            new_state.push(move)  # Apply the move

            # Check if this state is already explored
            if not any(child.move == move for child in node.children):
                new_node = MCTSNode(new_state, parent=node, move=move)
                node.children.append(new_node)
                return new_node

        return node  # Return the node if no expansion was done

    def _simulate(self, node):
        # Perform a random playout using get_best to simulate moves
        state = node.state.copy()
        while not state.is_checkmate() and not state.is_game_over()  and sum(1 for _ in state.legal_moves) > 0:
            best_move = self.get_best_function(state)  # Use the CNN to predict the best move
          #  print(best_move, chess.Move.from_uci(best_move), state.legal_moves)
            if chess.Move.from_uci(best_move) not in state.legal_moves:
             #   print("Best move is illegal, choosing a random legal move.")
                best_move = random.choice(list(state.legal_moves)).uci()
            state.push_uci(best_move)
        return state.result()  # Return the result: '1-0' for white win, '0-1' for black win, '1/2-1/2' for draw

    def _backpropagate(self, node, winner):
        # Backpropagate the result of the simulation up the tree
        while node is not None:
            node.visits += 1
            if winner == '1-0':  # White wins
                node.value += 1
            elif winner == '0-1':  # Black wins
                node.value -= 1
            else:  # Draw
                node.value += 0.5
            node = node.parent


# Example usage:
# Assuming we have a chess board state (from the `chess` library) and the `get_best` function
def get_best_move(board):
    # Your CNN function that predicts the best move for the given board
    return predict_chess_move(board.fen(),model, device)



def chessmarro_mcts_predict_chess_move(fen, simulations=10):
# Set up the initial chess board state
    board = chess.Board(fen)

    # Create the root node
    root = MCTSNode(state=board)

    # Initialize MCTS with n simulations
    mcts = MCTS(root=root, get_best_function=get_best_move, simulations=simulations)

    # Run the MCTS algorithm and get the best move
    best_move_node = mcts.search()
    best_move = best_move_node

    # Display the best move
    print(f"The best move predicted is: {best_move}")
    return best_move