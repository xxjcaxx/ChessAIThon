import queue
import multiprocessing as mp
from threading import Lock
import threading
import chess
import random
import math
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gc

def normalize_moves(moves):
    total = sum(score for _, score in moves)
    if total <= 0:
        n = len(moves)
        return [(m, 1.0 / n) for m, _ in moves]
    return [(m, score / total) for m, score in moves]

def filter_by_cumulative_probability(moves, threshold=0.80, min_k=3):
    #moves = prediction['moves'] # Ya vienen ordenadas
    filtered = []
    cumulative = 0
    
    for i, (move, score) in enumerate(moves):
        filtered.append((move, score))
        cumulative += score
        # Paramos si superamos el umbral, pero siempre exploramos al menos min_k
        if cumulative >= threshold and i >= (min_k - 1):
            break
            
    return normalize_moves(filtered)

def apply_dirichlet_noise(moves_with_scores, epsilon=0.35, alpha=0.2):
    if not moves_with_scores:
        return []
    
    n = len(moves_with_scores)
    # Generamos el ruido basado en la distribución de Dirichlet
    noise = np.random.dirichlet([alpha] * n)
    
    new_moves = []
    for i, (move, score) in enumerate(moves_with_scores):
        # Mezclamos la opinión de la red con el ruido
        adjusted_score = (1 - epsilon) * score + epsilon * noise[i]
        new_moves.append((move, adjusted_score))
        
    return new_moves

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
        
        sqrt_total_visits = math.sqrt(self.visits + 1)

        # Calculamos un valor de referencia para nodos no visitados (FPU)
        # Si ya tenemos hijos visitados, usamos el promedio de sus Q. 
        # Si no, usamos un valor base (puedes probar con 0 o -0.1)
        v_visited = [c.value / c.visits for c in self.children if c.visits > 0]
        fpu_value = sum(v_visited) / len(v_visited) if v_visited else 0.0

        for child in self.children:
            with child.lock:
                # 1. Q: Explotación
                actual_visits = child.visits + child.vloss
                if actual_visits > 0:
                    q_value = (child.value - child.vloss) / actual_visits
                else:
                    # First Play Urgency: Valor asignado a nodos no explorados
                    q_value = fpu_value 

                # 2. U: Exploración (Fórmula PUCT estándar de AlphaZero)
                u_value = c_puct * child.prior_p * (sqrt_total_visits / (1 + actual_visits))
                
                score = q_value + u_value
                
                if score > best_value:
                    best_value = score
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
        # 1. Obtenemos la predicción completa (Diccionario) de la raíz
        prediction = get_best_function(root_state)
        
        # 2. Extraemos y normalizamos las jugadas
        moves_with_scores = filter_by_cumulative_probability(prediction['moves'])

        # 2. Inyectamos Ruido (SOLO en la raíz)
        moves_with_noise = apply_dirichlet_noise(moves_with_scores)
        
        # 3. Extraemos el valor de la red
        # Si el valor es relativo al que mueve, lo dejamos tal cual
        root_nn_value = prediction['value']
        
        #print(f"Evaluación inicial de la red:  {moves_with_noise[0][0]} {root_nn_value:.3f}")
        #print(f"Evaluación inicial de la red: {root_nn_value:.3f}",f" Mejor jugada inicial: {moves_with_noise}")

        # 4. Creamos la raíz e inyectamos el valor inicial
        self.root = MCTSNode(root_state, priority_moves=moves_with_noise)
       
        self.get_best_function = get_best_function
        self.simulations = simulations
        self.puct = puct

    def _apply_vloss(self, path):
        for node in path:
            with node.lock:
                node.vloss += 1

    def _remove_vloss(self, path):
        for node in path:
            with node.lock:
                node.vloss -= 1
    
    def multi_thread_run(self, num_simulations):
        max_d = 0
        total_d = 0
        for _ in range(num_simulations):
            depth = self.thread_search()
            if depth > max_d:
                max_d = depth
            total_d += depth
            
        return max_d, total_d / num_simulations if num_simulations > 0 else 0

    def thread_search(self):
        path = []
        node = self.root
        path.append(node)
        
        # 1. SELECCIÓN: Bajamos mientras el nodo esté totalmente expandido
        while node.is_fully_expanded() and not node.state.is_game_over():
            next_node = node.best_child(self.puct)
            if next_node is None: break
            node = next_node
            path.append(node)

        current_depth = len(path)
        self._apply_vloss(path)

        try:
            if not node.state.is_game_over():
                move_to_try = None
                with node.lock:
                    if node.untried_moves:
                        move_to_try = node.untried_moves.pop()

                if move_to_try:
                    move_str, score = move_to_try
                    new_state = node.state.copy()
                    new_state.push(chess.Move.from_uci(move_str))
                    
                    # Petición a la GPU FUERA DEL LOCK
                    prediction = self.get_best_function(new_state)
                    
                    #child_moves = normalize_moves(prediction['moves'])
                    # Filtramos los que mayor probabilidad acumulen
                    child_moves = filter_by_cumulative_probability(prediction['moves'])

                    child_value = prediction['value']
                    
                    new_node = MCTSNode(new_state, parent=node, move=move_str, 
                                        priority_moves=child_moves, prior_p=score)
                    
                    with node.lock:
                        node.children.append(new_node)
                    
                    self._backpropagate(new_node, child_value)
                else:
                        # Si llegamos aquí y no hay untried_moves pero tampoco es game_over,
                        # es que ya estaba expandido. Hacemos backpropagate del valor conocido.
                        # (Opcional: puedes pedir una nueva evaluación si quieres)
                        self._backpropagate(node, 0) 
        finally:
            self._remove_vloss(path)
        return current_depth

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

        # 1. Llamamos a la red para la raíz
        prediction = get_best_function(new_state)
        
        # 2. Extraemos y normalizamos solo las jugadas
        # prediction['moves'] es [(uci, score), (uci, score)...]
        priority_moves = normalize_moves(prediction['moves'])

        #priority_moves = normalize_moves(self.get_best_function(new_state))
        new_node = MCTSNode(
            new_state,
            parent=node,
            move=move,
            priority_moves=priority_moves,
            prior_p=score
        )

        node.children.append(new_node)
        return new_node

    

    def _backpropagate(self, node, value):
        # value viene desde la perspectiva del jugador en 'node'.
        # Queremos subir al padre. El padre es el oponente.
        # Por tanto, el valor para el padre debe ser invertido.
        while node is not None:
            with node.lock:
                node.visits += 1
                node.value += value
            
            value = -value # Invertimos para el siguiente nivel (el padre)
            node = node.parent



def mcts_worker_persistent(batcher_q, mcts_result_q, worker_response_queue, task_in_q, id, puct):
    n_threads = 4
    

    local_q = queue.Queue()
   ## thread_responses = {}
    thread_responses = {i: queue.Queue() for i in range(1000)} # Diccionario pre-asignado o gestión fija
    # Nota: threading.get_ident() puede dar IDs muy altos, mejor mapear t_id a un índice 0..n_threads
    SENTINEL = None
    executor = ThreadPoolExecutor(max_workers=n_threads)
   


    def mirror_uci_move(uci_str):
        move = chess.Move.from_uci(uci_str)
        # Refleja el movimiento verticalmente
        mirrored_move = chess.Move(
            from_square=chess.square_mirror(move.from_square),
            to_square=chess.square_mirror(move.to_square),
            promotion=move.promotion
        )
        return mirrored_move.uci()


    def get_best_move(board):
        t_id = threading.get_ident()
        if t_id not in thread_responses:
            thread_responses[t_id] = queue.Queue()
        
        # --- PASO 1: Inversión de perspectiva ---
        # Si mueven negras, enviamos el tablero espejo (mirror)
        is_black = (board.turn == chess.BLACK)
        fen_to_send = board.mirror().fen() if is_black else board.fen()
        
        local_q.put((t_id, fen_to_send))
        response = thread_responses[t_id].get() 
        
        # --- PASO 2: Inversión de la respuesta ---
        if is_black:
            # 1. Invertimos las jugadas (ej: e7e5 se convierte en e2e4)
            moves_mirrored = [
                (mirror_uci_move(m), s) for m, s in response['moves']
            ]
            # 2. El valor suele ser relativo al que mueve. 
            # Si la red dice 0.8 para el tablero invertido de las negras, 
            # significa que las NEGRAS están +0.8. 
            # No necesitas invertir el signo si tu MCTS ya espera el valor 
            # relativo al jugador actual.
            response['moves'] = moves_mirrored

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

    try:
        while True:

            task = task_in_q.get() # Espera a que llegue el FEN
            if task is None: 
                break # Señal de apagado

            thread_responses.clear()

            #print(f"Worker {id} lanzando {n_threads} hilos... {task[2]} simulaciones")

            fen = task[1]
            simulations = task[2]

            board = chess.Board(fen)
            #print("Initial board for MCTS:\n", board)
            #root = MCTSNode(state=board)
            mcts = MCTS(root_state=board, get_best_function=get_best_move, simulations=simulations, puct=puct)

            #with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Repartimos las simulaciones totales entre los hilos
            sims_per_thread = simulations // n_threads
            futures = [executor.submit(mcts.multi_thread_run, sims_per_thread) 
                                for _ in range(n_threads)]
            results = [f.result() for f in futures]

            worker_max_depth = max(r[0] for r in results)
            worker_avg_depth = sum(r[1] for r in results) / len(results)

            #print(f"Worker {id} [PUCT {puct:.1f}]: Max Depth: {worker_max_depth}, Avg Depth: {worker_avg_depth:.2f}")

            with mcts.root.lock:
                best_child = max(mcts.root.children, key=lambda c: c.visits)
                best_move = best_child.move
                all_moves = [(str(c.move), c.visits) for c in mcts.root.children]

            #local_q.put(SENTINEL)
            #worker_response_queue.put(SENTINEL)
                
            mcts_result_q.put((id, best_move, all_moves))
            #print("MCTS worker finished", id)
            thread_responses.clear()
            def clear_tree(node):
                for child in node.children:
                    clear_tree(child)
                node.children = []
                node.parent = None  # Rompemos la referencia circular

            clear_tree(mcts.root)
            
            del mcts
            del board
            gc.collect()
    finally:
        executor.shutdown(wait=False)
        # Enviar sentinels para cerrar sender/receiver
        local_q.put(None)

    sender_thread.join()
    receiver_thread.join()

    
