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
import os
import psutil
from chess_aux_optim import  concat_fen_legal
import torch

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
        
    #print(new_moves)
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
        self.predicted_value = None  # Valor predicho por la red al expandir este nodo
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
        self.get_best_function = get_best_function
        self.simulations = simulations
        self.puct = puct
        
        # 1. Obtenemos la predicción completa (Diccionario) de la raíz
        prediction = self._evaluate_state(root_state)
        
        
        # 2. Extraemos y normalizamos las jugadas
        moves_with_scores = filter_by_cumulative_probability(prediction['moves'])

        # 2. Inyectamos Ruido (SOLO en la raíz)
        moves_with_noise = apply_dirichlet_noise(moves_with_scores)
       
        print("🌳 Root moves after noise (top 10):\n", [(m, f"{s:.3f}") for m, s in moves_with_noise[:10]])
        
        self.initial_moves = list(prediction['moves'])  # Guardamos esto para análisis o visualización futura

        # 4. Creamos la raíz e inyectamos el valor inicial
        self.root = MCTSNode(root_state, priority_moves=moves_with_noise)
        for move_str, prior_p in self.root.untried_moves[:]:
            new_state = self.root.state.copy()
            new_state.push(chess.Move.from_uci(move_str))
            
            # Evalúa con la red (single-threaded)
            prediction = self.get_best_function(new_state)
            child_moves = filter_by_cumulative_probability(prediction['moves'])
            child_value = prediction['value']
            
            new_node = MCTSNode(new_state, parent=self.root, move=move_str, 
                                priority_moves=child_moves, prior_p=prior_p)
            new_node.predicted_value = child_value
            
            self.root.children.append(new_node)
        # Ahora todos los movimientos de la raíz están expandidos
        self.root.untried_moves = []  # Ya no queda nada por explorar


    def _apply_vloss(self, path):
        for node in path:
            with node.lock:
                node.vloss += 1

    def _remove_vloss(self, path):
        for node in path:
            with node.lock:
                node.vloss -= 1

    def _evaluate_state(self, state):
        """Evalúa un estado, manejando la perspectiva de negras correctamente.
        La red está entrenada solo con el turno de las blancas, así que:
        1. Si es turno de negras, invertimos el tablero (mirror)
        2. Llamamos a la red con el tablero invertido
        3. Invertimos los movimientos de la respuesta
        """
        is_black = (state.turn == chess.BLACK)
        state_to_eval = state.mirror() if is_black else state
        
        prediction = self.get_best_function(state_to_eval)
        
        if is_black:
            # Invertir los movimientos
            moves_mirrored = []
            for move_uci, score in prediction['moves']:
                move = chess.Move.from_uci(move_uci)
                mirrored = chess.Move(
                    from_square=chess.square_mirror(move.from_square),
                    to_square=chess.square_mirror(move.to_square),
                    promotion=move.promotion
                )
                moves_mirrored.append((mirrored.uci(), score))
            prediction['moves'] = moves_mirrored
           
            # 2. ¡CLAVE!: Invertir el signo del VALUE
            # Si la red dice -0.893 para las negras, devolvemos 0.893
            # para que el backpropagate lo asigne correctamente al padre.
            #prediction['value'] = -prediction['value']
        
        return prediction
    
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
            if next_node is None: 
               # print("Warning: best_child returned None during selection. This should not happen if the tree is properly expanded.")
                break
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
                    
                    prediction = self._evaluate_state(new_state)
                    child_moves = filter_by_cumulative_probability(prediction['moves'])
                    child_value = prediction['value']

                    new_node = MCTSNode(new_state, parent=node, move=move_str, 
                                        priority_moves=child_moves, prior_p=score)
                    new_node.predicted_value = child_value
                    
                    with node.lock:
                        node.children.append(new_node)
                    
                    self._backpropagate(new_node, -child_value)
                else:
                        # Si llegamos aquí y no hay untried_moves pero tampoco es game_over,
                        # es que ya estaba expandido. Hacemos backpropagate del valor conocido.
                        # (Opcional: puedes pedir una nueva evaluación si quieres)
                        self._backpropagate(node, 0) 
            if node.state.is_game_over():
                #print("Terminal node reached at depth", current_depth)
                result = node.state.result()

                if result == "1-0":
                    # Ganan blancas
                    terminal_value = 1 if node.state.turn == chess.WHITE else -1

                elif result == "0-1":
                    # Ganan negras
                    terminal_value = 1 if node.state.turn == chess.BLACK else -1
                else:
                    terminal_value = 0
                self._backpropagate(node, terminal_value)
        finally:
            self._remove_vloss(path)
        return current_depth

    

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
    n_threads = 64
    

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
               # print("Worker", id, "sending item to batcher:", item)
                #item_tensor = concat_fen_legal_packed(item[1]) 
                item_tensor = concat_fen_legal(item[1])
                #print("Worker", id, "sending item to batcher:", item_tensor.shape, item_tensor.dtype)
                batcher_q.put((id, (item[0], item_tensor)), timeout=1.0)
                #batcher_q.put((id, item), timeout=1.0)
    
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
            
            # Verificar si el FEN está en jaque mate
            #print("checkmate:", board.is_checkmate())
            #print("stalemate:", board.is_stalemate())
            #print("legal moves:", list(board.legal_moves))
            if board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material():
                print(f"\033[1;31m⚠️  Worker {id}: FEN en JAQUE MATE - retornando None\033[0m")
                mcts_result_q.put((id, None, [], []))
                thread_responses.clear()
                continue
            
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

            print(f"Worker {id} [PUCT {puct:.1f}]: Max Depth: {worker_max_depth}, Avg Depth: {worker_avg_depth:.2f}")
            print("Root visits:", mcts.root.visits)
            print("Sum children:", sum(c.visits for c in mcts.root.children))

            with mcts.root.lock:
                best_child = max(mcts.root.children, key=lambda c: c.visits)
                best_move = best_child.move
                all_moves = [(str(c.move), c.visits) for c in mcts.root.children]
                ###########################3
                initial_moves = mcts.initial_moves  # Movimientos iniciales con ruido y filtrados
                #print(f"Worker {id} initial moves: {initial_moves}")
                # Imprimir valores predichos por la red para hijos de primer nivel
                if(id==0): # Solo lo imprimimos para el primer worker para no saturar la consola
                    try:
                        vals = []
                        for c in mcts.root.children:
                            pv = c.predicted_value if hasattr(c, 'predicted_value') else None
                            if pv is not None:
                                if pv > 0:
                                    color = "\033[92m"  # verde
                                elif pv < 0:
                                    color = "\033[91m"  # rojo
                                else:
                                    color = "\033[94m"  # azul para 0
                                vals.append(f"{c.move}:{color}{pv:.3f}\033[0m")
                            else:
                                vals.append(f"{c.move}:None")

                        print("🔮 Root children predicted values: ", " | ".join(vals))
                    except Exception:
                        pass

            #local_q.put(SENTINEL)
            #worker_response_queue.put(SENTINEL)
            #print(all_moves)
            mcts_result_q.put((id, best_move, all_moves, initial_moves))
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

            """def get_process_memory():
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return mem_info.rss / (1024 ** 2)  # Convertir a MB

            print(f"Consumo del proceso actual: {get_process_memory():.2f} MB")"""
    finally:
        executor.shutdown(wait=False)
        # Enviar sentinels para cerrar sender/receiver
        local_q.put(None)

    sender_thread.join()
    receiver_thread.join()

    
