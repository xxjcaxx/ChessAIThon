import queue
import multiprocessing as mp
from threading import Lock
import threading
import chess
import random
import math
import json
from concurrent.futures import ThreadPoolExecutor

def normalize_moves(moves):
    total = sum(score for _, score in moves)
    if total <= 0:
        n = len(moves)
        return [(m, 1.0 / n) for m, _ in moves]
    return [(m, score / total) for m, score in moves]

class MCTSNode:
    def __init__(self, state, parent=None, move=None, priority_moves=None, prior_p=0.0):
        self.state = state  # Current board state
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this state
        self.children = []  # Child nodes (future possible states)
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Total reward (win/loss/draw) from simulations
        self.prior_p = prior_p  # El score que la red le dio a esta jugada
        # Guardamos los movimientos que get_best_function nos dio.
        # Si priority_moves es [(move1, score1), (move2, score2)...]
        self.untried_moves = priority_moves if priority_moves is not None else []
        self.vloss = 0  # Contador de pérdida virtual
        self.lock = threading.Lock() # Para proteger cambios en el nodo
        

    def is_fully_expanded(self):
        # Returns True if all possible moves have been explored
        return len(self.untried_moves) == 0


    def best_child(self, c_puct=1.4):
        best_value = -float('inf')
        best_node = None
        
        # Determinamos si el jugador en este estado busca maximizar o minimizar
        # Si state.turn == chess.WHITE, significa que el movimiento hacia este nodo lo hizo NEGRO.
        # O más simple: el jugador que va a mover en este nodo es quien decide.
        is_white_to_move = self.state.turn == chess.WHITE
        # Raíz de las visitas totales para el factor de exploración
        sqrt_total_visits = math.sqrt(self.visits + 1)

        for child in self.children:

            with child.lock:
                # Calculamos Q restando la pérdida virtual
                # (child.visits + child.vloss) asegura que el denominador suba
                # (child.value - child.vloss) asegura que el valor baje
                v_loss_penalty = child.vloss * 1.0 # Penalización ajustable
                # 1. Q: El valor promedio (explotación)
            # Valor relativo: si es turno de negras, el valor se invierte
                q_value = (child.value - v_loss_penalty) / (child.visits + child.vloss + 1)
                  # 2. U: El factor de confianza/priors (exploración inteligente)
            # Usamos el prior_p (score de la red)
                u_value = c_puct * child.prior_p * (sqrt_total_visits / (1 + child.visits + child.vloss))
                
            puct_score = q_value + u_value
            if puct_score > best_value:
                best_value = puct_score
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
    def __init__(self, root_state, get_best_function, simulations=100, puct=1.4):
        #print("Initializing MCTS")
        moves_with_scores = normalize_moves(get_best_function(root_state)) 
        print("Initial moves with scores:", moves_with_scores[0])
        #print("All legal moves:", [move.uci() for move in root_state.legal_moves])


        self.root = MCTSNode(root_state, priority_moves=moves_with_scores)
        # 2. EXPANDIR LA RAÍZ INMEDIATAMENTE
        # Esto garantiza que root.children nunca esté vacío cuando los hilos terminen
        for move_str, score in moves_with_scores:
            new_state = root_state.copy()
            new_state.push(chess.Move.from_uci(move_str))
            child_node = MCTSNode(new_state, parent=self.root, move=move_str, prior_p=score)
            self.root.children.append(child_node)

        # Marcar como expandido
        self.root.untried_moves = []

        self.get_best_function = get_best_function  # Function to get the best move
        self.simulations = simulations  # Number of MCTS simulations
        self.puct = puct  # PUCT exploration constant
        #print("MCTS initialized with root state:", root_state.fen(), self.root.untried_moves) 

    def _apply_vloss(self, path):
        for node in path:
            with node.lock:
                node.vloss += 1

    def _remove_vloss(self, path):
        for node in path:
            with node.lock:
                node.vloss -= 1
    
    def multi_thread_run(self, num_simulations):
        for _ in range(num_simulations):
            self.thread_search()

    def thread_search(self):
        path = []
        node = self.root
        path.append(node)
        
        while node is not None and node.is_fully_expanded() and not node.state.is_game_over():
            next_node = node.best_child(self.puct)
            if next_node is None: # Si no hay hijos disponibles, nos quedamos en el actual
                break
            node = next_node
            path.append(node)
    
        if node is None: return # Seguridad extrema
        
        self._apply_vloss(path)

        try:
            if not node.state.is_game_over():
                # BLOQUEO DE EXPANSIÓN: Solo un hilo expande el nodo
                with node.lock:
                    needs_expansion = (not node.children and not node.untried_moves)
                
                if needs_expansion:
                    # El hilo libera el lock del nodo mientras espera a la GPU (get_best_move)
                    # Esto permite que otros hilos sigan usando el árbol
                    moves_with_scores = normalize_moves(self.get_best_function(node.state))
                    
                    with node.lock:
                        node.untried_moves = moves_with_scores
                        # Expandimos inmediatamente un hijo para que el nodo ya no cuente como "hoja"
                        if node.untried_moves:
                            move_str, score = node.untried_moves.pop()
                            new_state = node.state.copy()
                            new_state.push(chess.Move.from_uci(move_str))
                            
                            # El nuevo nodo heredará los parámetros de búsqueda
                            new_node = MCTSNode(new_state, parent=node, move=move_str, prior_p=score)
                            node.children.append(new_node)
                            node = new_node # Evaluamos el nuevo hijo
                            path.append(node)

            score = self._evaluate_position(node.state)
            self._backpropagate(node, score)
        finally:
            self._remove_vloss(path)

    def search(self):
        if not self.root.untried_moves:
        # No hay jugadas legales
            return None
        for _ in range(self.simulations):
            #print("Simulation", _+1)
            # Step 1: Selection
            node = self._select(self.root)
            #print("Selected node with move:", node.move, "Visits:", node.visits, "Value:", node.value, "Untried moves:", [m[0] for m in node.untried_moves])

            # Step 2: Expansion
            if not node.is_fully_expanded() and not node.state.is_game_over():
                node = self._expand(node)
                #print("Expanded to node with move:", node.move)
            # Step 3: Simulation (Playout)
            #winner = self._simulate(node)
            # En lugar de un bucle de 60 jugadas, evaluamos la posición del nodo actual
            score = self._evaluate_position(node.state)
            #print("Evaluated position score:", score)

            # Step 4: Backpropagation
            self._backpropagate(node, score)

        # Return the best move after simulations
        #print("Children:", len(self.root.children[0].children))
        #print(self.root.to_json())
        #for c in self.root.children:
           # print(c.move, c.visits, c.value,  c.value / c.visits)
        return max(self.root.children, key=lambda c: c.visits).move , [(c.move, c.visits) for c in self.root.children]

    def _select(self, node):
        # Traverse down the tree to find a leaf node
        if node.state.is_game_over():
            return node
        while node.is_fully_expanded() and not node.state.is_game_over():
            node = node.best_child(self.puct)
        return node

    def _expand(self, node):
        if node.state.is_game_over():
            return node

        if not node.untried_moves:
            return node

        idx = random.randrange(len(node.untried_moves))
        move, score = node.untried_moves.pop(idx)

        new_state = node.state.copy()
        move_obj = chess.Move.from_uci(move)

        if node.state.is_capture(move_obj):
            captured = node.state.piece_at(move_obj.to_square)
            if captured and captured.piece_type == chess.KING:
                return node  # o continue / skip

        new_state.push(move_obj)

        priority_moves = normalize_moves(self.get_best_function(new_state))
        new_node = MCTSNode(
            new_state,
            parent=node,
            move=move,
            priority_moves=priority_moves,
            prior_p=score
        )

        node.children.append(new_node)
        return new_node

    def _evaluate_position(self, state):
        """Evaluación estática simple pero informativa (blancas +, negras -)."""

        # 1. Estados terminales
        if state.is_checkmate():
            return 1.0 if state.turn == chess.BLACK else -1.0
        if state.is_game_over():
            return 0.0

        score = 0.0

        # 2. Material
        piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.1,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }

        for piece_type, val in piece_values.items():
            score += len(state.pieces(piece_type, chess.WHITE)) * val
            score -= len(state.pieces(piece_type, chess.BLACK)) * val

        # 3. Desarrollo (solo piezas menores)
        development_squares = {
            chess.WHITE: {chess.C1, chess.F1, chess.B1, chess.G1},
            chess.BLACK: {chess.C8, chess.F8, chess.B8, chess.G8},
        }

        for color in [chess.WHITE, chess.BLACK]:
            undeveloped = 0
            for sq in development_squares[color]:
                piece = state.piece_at(sq)
                if piece and piece.color == color:
                    undeveloped += 1
            score += (-0.1 * undeveloped) if color == chess.WHITE else (0.1 * undeveloped)

        # 4. Enroque / seguridad del rey
        for color in [chess.WHITE, chess.BLACK]:
            king_sq = state.king(color)
            if king_sq is None:
                continue

            if color == chess.WHITE:
                if king_sq in [chess.G1, chess.C1]:
                    score += 0.15
            else:
                if king_sq in [chess.G8, chess.C8]:
                    score -= 0.15

        # 5. Control del centro (d4, e4, d5, e5)
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for sq in center_squares:
            attackers_white = state.attackers(chess.WHITE, sq)
            attackers_black = state.attackers(chess.BLACK, sq)
            score += 0.05 * (len(attackers_white) - len(attackers_black))

        # 6. Movilidad simple (número de jugadas legales)
        mobility = len(list(state.legal_moves))
        score += 0.002 * (mobility - 30)

        # 7. Normalización suave
        return max(-1.0, min(1.0, score / 4.0))



    def _backpropagate(self, node, value):
        while node is not None:
            with node.lock: # Bloqueo rápido para evitar condiciones de carrera
                node.visits += 1
                node.value += value
            value = -value # Alternar signo según el turno
            node = node.parent




def mcts_worker(batcher_q, mcts_result_q, worker_response_queue, id, task, puct):
    n_threads = 4
    print(f"Worker {id} lanzando {n_threads} hilos...")

    local_q = queue.Queue()
    thread_responses = {}
    SENTINEL = None
    fen = task[1]
    simulations = task[2]
   

    """def get_best_move(board):
        #print("Best move init:", board.fen())
        # Your CNN function that predicts the best move for the given board
        local_q.put((0, board.fen()))
        response = thread_responses[0].get()
        #print("Best move received:", response)
        return response"""

    def get_best_move(board):
        #print(board)
        t_id = threading.get_ident()
        
        # Si el hilo no tiene cola, la crea (sucederá la primera vez que cada hilo pida algo)
        if t_id not in thread_responses:
            thread_responses[t_id] = queue.Queue()
        
        # Enviamos la petición indicando el ID del hilo
        local_q.put((t_id, board.fen()))
        #print((t_id, board.fen()))
        # Esperamos en la cola específica de este hilo
        response = thread_responses[t_id].get()
        #print(response)
        return response



    #print("MCTS worker running simulations...", id)
    def sender():
        while True:
            item = local_q.get()
            if item is SENTINEL:
                break
            try:
                #print("Worker", id, "sending item to batcher:", item)
                batcher_q.put((id, item), timeout=1.0)
            except queue.Full:
                print(f"Worker {id} bloqueado: batcher_q está llena")

    def receiver():
        
        while True:
            
            response = worker_response_queue.get()
            #print("MCTS worker"+str(id)+" received response:", response)
            if response is SENTINEL: break
            
            identificadores, data = response
            t_id = identificadores[1][0]  # Extraemos el id del thread
            # Entregamos el resultado al thread que lo pidió
            if t_id in thread_responses:
                thread_responses[t_id].put(data)            

    

    sender_thread = threading.Thread(target=sender)
    sender_thread.start()
    receiver_thread = threading.Thread(target=receiver)
    receiver_thread.start()


    board = chess.Board(fen)
    #print("Initial board for MCTS:\n", board)
    #root = MCTSNode(state=board)
    mcts = MCTS(root_state=board, get_best_function=get_best_move, simulations=simulations, puct=puct)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Repartimos las simulaciones totales entre los hilos
        futures = [executor.submit(mcts.multi_thread_run, simulations // n_threads) 
                   for _ in range(n_threads)]
        for f in futures: f.result()

    with mcts.root.lock:
        best_child = max(mcts.root.children, key=lambda c: c.visits)
        best_move = best_child.move
        all_moves = [(str(c.move), c.visits) for c in mcts.root.children]

    local_q.put(SENTINEL)
    worker_response_queue.put(SENTINEL)
    sender_thread.join()
    receiver_thread.join()

    
        
    mcts_result_q.put((id, best_move, all_moves))
    print("MCTS worker finished", id)