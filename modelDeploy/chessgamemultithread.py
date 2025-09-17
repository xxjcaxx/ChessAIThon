import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from chessmodel import init_model, predict_chess_move
import json
import sys

sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr

import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory



# mp.set_start_method('spawn', force=True)

def predict_chess_moves_vectorized(boards_tensor, temperature=1.2):
    """
    Predice movimientos para un batch de posiciones con Conv2D, usando
    movimientos legales codificados en los últimos 64 canales/elementos.
    """
    B = boards_tensor.size(0)
    device = boards_tensor.device

    with torch.no_grad():
        outputs = model(boards_tensor)  # [B, 4096], conv2d espera [B, C, H, W]

    # Extraer máscaras legales de los últimos 64 canales
    # Flatten del canal y el tablero a 4096
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).to(dtype=torch.bool)

    # Poner -inf en posiciones ilegales
    masked_logits = torch.where(legal_masks, outputs, -float('inf'))

    # Softmax con temperatura
    probs = torch.softmax(masked_logits / temperature, dim=1)  # [B, 4096]

    # Samplear movimiento
    move_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

    # Convertir a UCI
    pred_moves = [number_to_uci(idx.item()) for idx in move_indices]

    return pred_moves


def batch_predict_worker(input_queue, output_queue, device='cuda:0'):
    """
    Worker que recibe batches de boards, construye el tensor completo y predice movimientos.
    """
    print("init worker")


    while True:
        item = input_queue.get()
        if item is None:
            print("item none")
            break  # Señal de cierre

        board_list = item  # Lista de tensores individuales [C, H, W]

        # Unir todos los tensores en un batch
        boards_tensor = torch.stack(item)  # [B,C,H,W] en CPU
        boards_tensor = boards_tensor.to(device)  # mover a GPU

        # Llamar a la función vectorizada de predicción
        pred_moves = predict_chess_moves_vectorized(boards_tensor)

        output_queue.put(pred_moves)

class ChessBatcher:
    """
    Clase para acumular posiciones y procesarlas en batch usando shared memory.
    """
    def __init__(self, batch_size=16, device='cuda:0'):
        self.batch_size = batch_size
        self.device = device
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.worker = Process(target=batch_predict_worker,
                              args=(self.input_queue, self.output_queue, device))
        self.worker.start()
        self.current_batch = []

    def add_board(self, board_tensor):
        """
        Añade un tensor de board individual al batch.
        """
        self.current_batch.append(board_tensor)

        if len(self.current_batch) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self):
        """
        Envía el batch acumulado al worker y limpia la lista.
        """
        if self.current_batch:
            self.input_queue.put(self.current_batch)
            self.current_batch = []

    def get_predictions(self):
        """
        Recupera todas las predicciones disponibles en la cola de salida.
        """
        predictions = []
        while not self.output_queue.empty():
            predictions.extend(self.output_queue.get())
        return predictions

    def close(self):
        """
        Cierra el worker correctamente.
        """
        self._flush_batch()
        self.input_queue.put(None)  # Señal de cierre
        self.worker.join()

"""

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
      #  Retorna el nodo y sus hijos (hasta profundidad depth) en JSON
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
    return predict_chess_move(board.fen(),model,device)

"""

def chessmarro_mcts_predict_chess_move(fen, simulations, model, device):
# Set up the initial chess board state
    #board = chess.Board(fen)

    # Create the root node
    #root = MCTSNode(state=board)

    # Initialize MCTS with n simulations
    #mcts = MCTS(root=root, get_best_function=get_best_move, simulations=simulations)

    # Run the MCTS algorithm and get the best move
    #best_move_node = mcts.search()
    #best_move = best_move_node
    best_move = 1234

    # Display the best move
    #print(f"The best move predicted is: {best_move}")


####################################3

    fens = [
        "r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8",
        "r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 2",
        '5k2/R7/3K4/4p3/5P2/8/8/5r2 w - - 0 0',
    '5k2/1R6/4p1p1/1pr3Pp/7P/1K6/8/8 w - - 0 0',
    '5k2/8/p7/4K1P1/P4R2/6r1/8/8 b - - 0 0',
    '8/8/8/p2r1k2/7p/PP1RK3/6P1/8 b - - 0 0',
    '8/8/8/1P4p1/5k2/5p2/P6K/8 b - - 0 0',
    '3b2k1/1p3p2/p1p5/2P4p/1P2P1p1/5p2/5P2/4RK2 w - - 0 0',
    '5k2/3R4/2K1p1p1/4P1P1/5P2/8/3r4/8 b - - 0 0',
    '6k1/6pp/5p2/8/5P2/P7/2K4P/8 b - - 0 0',
    '8/3R4/8/r3N2p/P1Pp1P2/2k2K1P/3r4/8 w - - 0 0',
    '6k1/8/6r1/8/5b2/2PR4/4K3/8 w - - 0 0',
    '8/1p3k2/3B4/8/3b2P1/1P6/6K1/8 b - - 0 0',
    '8/8/8/2p1k3/P6R/1K6/6rP/8 w - - 0 0',
    '6k1/5p1p/6p1/1P1n4/1K4P1/N6P/8/8 w - - 0 0',
    '8/k5r1/2N5/PK6/2B5/8/8/8 b - - 0 0',
    '6k1/8/5K2/8/5P1R/r6P/8/8 b - - 0 0',
    '8/8/4k1KP/p5P1/r7/8/8/8 w - - 0 0',
    '1R6/p2r4/2ppkp2/6p1/2PKP2p/P4P2/6PP/8 b - - 0 0',
    '8/7p/6p1/8/k7/8/2K3P1/8 b - - 0 0',
    'R7/8/8/6p1/4k3/3rPp1P/8/6K1 b - - 0 0',
    '8/7p/1p1k2p1/p1p2p2/8/PP2P2P/4KPP1/8 w - - 0 0' 
    ]

    # Preprocesar todas las posiciones
    boards = [concat_fen_legal(fen).cpu() for fen in fens]  # cada uno es numpy

    #print(boards)
    batcher = ChessBatcher(batch_size=8)

    # Supongamos que tienes una lista de tensores [C, H, W] por cada FEN
    for board_tensor in boards:
        batcher.add_board(board_tensor)

    # Asegúrate de enviar el batch final que quede
    batcher._flush_batch()

    # Recuperar predicciones
    moves = batcher.get_predictions()
    print(moves)

    # Cerrar worker
    batcher.close()


#######################################

    return best_move