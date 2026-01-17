import queue
import multiprocessing as mp
from threading import Lock
import threading
import chess
import random
import math
import json

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
            #print(_)
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

    def _simple_eval(self, board):
        """Lightweight material-count evaluation: positive => White advantage."""
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9}
        score = 0
        for piece_type, val in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * val
            score -= len(board.pieces(piece_type, chess.BLACK)) * val
        return score

    def _simulate(self, node):
        # Perform a random playout using get_best to simulate moves
        state = node.state.copy()
        DEPTH_CAP = 60
        depth = 0
        while not state.is_checkmate() and not state.is_game_over()  and sum(1 for _ in state.legal_moves) > 0:
            if (
                depth >= DEPTH_CAP
                or state.can_claim_fifty_moves()
                or state.can_claim_threefold_repetition()
            ):
                # --- Evaluate the truncated position ---
                eval_score = self._simple_eval(state)

                # --- Normalize the material difference to a bounded range (-1, 1) ---
                # A difference of ±10 (e.g., una dama entera) se considera máxima ventaja.
                normalized = max(-1.0, min(1.0, eval_score / 10.0))

                # --- Apply a small penalty for deeper (slower) simulations ---
                # Incentiva mates o ventajas más rápidas
                decay = 1.0 - min(0.5, depth / DEPTH_CAP * 0.5)
                normalized *= decay

                # --- Add a small neutral zone to avoid oscillations ---
                thresh = 0.05
                if normalized > thresh:
                    result = '1-0'
                elif normalized < -thresh:
                    result = '0-1'
                else:
                    result = '1/2-1/2'

                return result
            
            best_move = self.get_best_function(state)  # Use the CNN to predict the best move
          #  print(best_move, chess.Move.from_uci(best_move), state.legal_moves)
            if chess.Move.from_uci(best_move) not in state.legal_moves:
             #   print("Best move is illegal, choosing a random legal move.")
                best_move = random.choice(list(state.legal_moves)).uci()
            state.push_uci(best_move)
            depth += 1
            
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




def mcts_worker(batcher_q, mcts_result_q, worker_response_queue, id, task):
    print("MCTS worker started", id)

    local_q = queue.Queue()
    thread_responses = {t: queue.Queue() for t in range(5)}
    SENTINEL = None
    fen = task[1]
    simulations = task[2]

    """def simulation(thread_id):
        for i in range(simulations):
            local_q.put((thread_id, fen))  #fen
            result = thread_responses[thread_id].get()"""

    def get_best_move(board):
        # Your CNN function that predicts the best move for the given board
        local_q.put((0, board.fen()))
        response = thread_responses[0].get()
        #print("Best move received:", response)
        return response

    board = chess.Board(fen)
    root = MCTSNode(state=board)
    mcts = MCTS(root=root, get_best_function=get_best_move, simulations=simulations)

    
    def sender():
        while True:
            item = local_q.get()
            if item is SENTINEL:
                break
            try:
                batcher_q.put((id, item), timeout=1.0)
            except queue.Full:
                print(f"Worker {id} bloqueado: batcher_q está llena")

    def receiver():
        """Escucha respuestas globales y las reparte a los threads locales"""
        while True:
            # Se espera que el batcher devuelva (thread_id, result_data)
            response = worker_response_queue.get()
            #print("MCTS worker"+str(id)+" received response:", response)
            if response is SENTINEL: break
            
            identificadores, data = response
            t_id = identificadores[1][0]  # Extraemos el id del thread
            # Entregamos el resultado al thread que lo pidió
            if t_id in thread_responses:
                thread_responses[t_id].put(data)            

    threads = []

    sender_thread = threading.Thread(target=sender)
    sender_thread.start()
    receiver_thread = threading.Thread(target=receiver)
    receiver_thread.start()


    """for t in range(5):
        th = threading.Thread(target=simulation, args=(t,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()"""

    best_move = mcts.search()
    print("Best move from MCTS:", best_move)

    local_q.put(SENTINEL)
    worker_response_queue.put(SENTINEL)
    sender_thread.join()
    receiver_thread.join()

    
        
    mcts_result_q.put((id, best_move))
    print("MCTS worker finished", id)