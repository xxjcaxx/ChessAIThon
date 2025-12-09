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
import threading
import time

sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr

import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory
from queue import Empty
from multiprocessing import Manager
import concurrent.futures

# Enable cudnn autotuner for better conv performance on fixed-size inputs
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# Función central que realiza la inferencia en la GPU. 
# Procesa múltiples tableros a la vez (batching) para maximizar la eficiencia de la Conv2D.
def predict_chess_moves_vectorized(boards_tensor, temperature, model):
    # Get the batch size, which is the number of boards to process.
    B = boards_tensor.size(0)

    # Disable gradient calculations to save memory and computation time during inference.
    with torch.no_grad():
        # Pass the batch of board features through the neural network to get raw output logits.
        logits = model(boards_tensor)  # [B,4096]

    # --- LEGAL MASK ---
    # Extract the 64-channel legal move mask from the board tensor (assuming it's the last 64 channels).
    # Flatten the 64x8x8 mask into a 4096-element boolean vector for each board.
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).bool()

    # --- SANITIZE LOGITS  ---
    # Get the floating-point information for the logits' data type (e.g., float32).
    finfo = torch.finfo(logits.dtype)
    # Define a numerically stable, large negative value for masking illegal moves.
    negbig = finfo.min / 4
    # Replace any NaN, posinf, or neginf values in the logits to ensure numerical stability.
    logits = torch.nan_to_num(logits, nan=0.0,
                              # Clamp positive infinities to a large but stable number.
                              posinf=finfo.max/4,
                              # Clamp negative infinities to a small but stable number.
                              neginf=finfo.min/4)

    # --- MASK: Set illegal moves to a very low value ---
    # Set the logits for all illegal moves to the large negative value (`negbig`).
    masked = logits.masked_fill(~legal_masks, negbig)

    # --- SOFTMAX ---
    # Apply temperature scaling (controls exploration) and the softmax function to get move probabilities.
    probs = torch.softmax(masked / temperature, dim=1)

    # --- HANDLE INVALID ROWS (e.g., Checkmate/Stalemate) ---
    # Ensure all resulting probabilities are non-negative due to potential floating-point underflow.
    probs = torch.clamp(probs, min=0.0)
    # Calculate the sum of probabilities for each board in the batch.
    row_sums = probs.sum(dim=1, keepdim=True)
    # Identify boards where the probability sum is valid (greater than zero and finite).
    valid = (row_sums > 0) & torch.isfinite(row_sums)

    # If the probability sum is zero (invalid row), prepare a uniform distribution over all legal moves.
    # Create a float tensor from the legal mask for the uniform distribution fallback.
    uniform = legal_masks.float()
    # Normalize the uniform mask so that the probability sum of legal moves is exactly 1.
    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp(min=1)

    # For valid rows, use the normalized predicted probabilities; otherwise, use the uniform distribution fallback.
    final_probs = torch.where(valid, probs / row_sums, uniform)

        # --- SAMPLE (Standard MCTS Mode) ---
        # Draw one sample move index from the distribution for each board.
        # torch.multinomial requires a 2D tensor.
    idxs = torch.multinomial(final_probs, num_samples=1).squeeze(1)

        # --- Convert to UCI ---
        # Convert the selected move indices into Universal Chess Interface (UCI) string format.
        # The function 'number_to_uci' is assumed to be accessible.
    return [number_to_uci(int(i)) for i in idxs]

     
# Proceso secundario (Worker) que se ejecuta continuamente en la GPU. 
# Su única tarea es esperar batches en la input_queue, hacer la predicción rápida 
# y enviar los resultados a la output_queue.
def batch_predict_worker(input_queue, output_queue, model, device):
    """
    Worker que recibe batches de boards, construye el tensor completo y predice movimientos.
    """
    model = model.to(device)
    model.eval()

    while True:
        item = input_queue.get() # Bloquea y espera un nuevo batch
        if item is None:
            break # Señal de terminación

        boards, ids = item  # boards = [tensor], ids = [task_id]
        # Stack on CPU; if CUDA available, use pinned memory and non_blocking transfer
        boards_tensor = torch.stack(boards)
        if isinstance(device, str) and device.startswith("cuda") and boards_tensor.device.type == 'cpu':
            try:
                boards_tensor = boards_tensor.pin_memory()
            except Exception:
                pass
            # Move to device using non_blocking to overlap copies
            to_kwargs = {'non_blocking': True}
            boards_tensor = boards_tensor.to(device, **to_kwargs)
        else:
            boards_tensor = boards_tensor.to(device)

        # Use mixed precision on CUDA for faster inference
        if isinstance(device, str) and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                preds = predict_chess_moves_vectorized(boards_tensor, 1.2, model)  # lista de jugadas
        else:
            preds = predict_chess_moves_vectorized(boards_tensor, 1.2, model)  # lista de jugadas
        results = list(zip(ids, preds))  # [(task_id, pred), ...]  # Empareja el ID de tarea con la predicción.
        #print(results)
        output_queue.put(results) # Envía el resultado a la cola principal (host)


import uuid
from multiprocessing import Process, Queue
# Proceso secundario dedicado a recibir resultados del worker (output_queue) 
# y redirigirlos a la cola de respuesta específica de cada cliente (pending dictionary).
def dispatch_loop(output_queue,pending):
    print("displach loop")
    while True:
        results = output_queue.get(block=True) # Espera un resultado del worker
        for tid, pred in results:
            #print(tid,pred)
            if tid in pending:
                pending[tid].put(pred)  # Despacha la predicción a la cola del cliente original
                del pending[tid]  # Elimina la tarea pendiente


# Clase que implementa el patrón de Batching. Actúa como una interfaz para los clientes,
# acumulando peticiones hasta alcanzar el tamaño del batch (batch_size) antes de enviarlas a la GPU.
class ChessBatcher:
    """
    Clase para acumular posiciones y procesarlas en batch usando shared memory.
    Soporta múltiples clientes concurrentes y mantiene la correspondencia
    entre petición y predicción.
    """
    def __init__(self, batch_size, model, device, manager, flusher_interval=0.2):
        self.batch_size = batch_size
        self.device = device
        self.input_queue = Queue() # Cola para enviar datos al worker (CPU -> GPU Worker)
        self.output_queue = Queue() # Cola para recibir datos del worker (GPU Worker -> CPU)
        # Proceso Worker: Ejecuta el worker en paralelo.
        self.worker = Process(target=batch_predict_worker,
                              args=(self.input_queue, self.output_queue, model, device))
        self.worker.start()
        self.current_batch = []
        # Lock to protect current_batch between flusher thread and add_board
        self._flush_lock = threading.Lock()
        # Counters to measure flush behavior
        self.flush_by_size_count = 0
        self.flush_by_timer_count = 0
        # Counters for batch statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0
    # Track max batch size and distribution (small histogram)
        self.max_batch_size = 0
        self.batch_size_counts = {}
    # Event to stop the flusher cleanly
        self._stop_event = threading.Event()
        # Start background flusher thread to collect small requests into batches
        # Allow the flusher interval to be tuned (defaults to 0.1s)
        self._flusher_interval = flusher_interval
        self._flusher_thread = threading.Thread(target=self._flusher_loop, daemon=True)
        self._flusher_thread.start()

        self.manager = manager
            # Diccionario compartido: Almacena las colas de respuesta individuales por ID de tarea.
        self.pending = self.manager.dict()  # task_id -> Queue de respuesta
            
            # Proceso Dispatch: Mueve los resultados de la output_queue a la cola individual del cliente.
        self.dispatch_loop_process = Process(target=dispatch_loop, args=(self.output_queue,self.pending), daemon=True)
        self.dispatch_loop_process.start()


    def add_board(self, board_tensor, response_q=None):
         # Genera una cola de respuesta si no se proporciona una.
        if response_q is None:
            response_q = self.manager.Queue()
        task_id = uuid.uuid4().hex # Genera un ID único para rastrear la petición.
        self.pending[task_id] = response_q # Asocia el ID a la cola de respuesta.
        # Append under lock; don't call _flush_batch while holding the lock (avoid reentrant lock)
        should_flush = False
        with self._flush_lock:
            self.current_batch.append((board_tensor, task_id)) # Acumula la petición.
            self.total_requests += 1
            # Verifica si se ha alcanzado el tamaño del batch.
            if len(self.current_batch) >= self.batch_size:
                should_flush = True
        if should_flush:
            self._flush_batch(caller='size') # Si es así, envía el lote a la GPU.
        return response_q # Devuelve la cola donde el cliente esperará la respuesta

    def _flush_batch(self, caller=None):
        """
        Envía el batch acumulado al worker y limpia la lista.
        """
        # Caller should hold _flush_lock when appropriate; double-check here
        with self._flush_lock:
            if self.current_batch:
                boards, ids = zip(*self.current_batch)
                self.input_queue.put((boards, ids))
                batch_len = len(ids)
                self.current_batch = []
                # Track flush origin
                if caller == 'size':
                    self.flush_by_size_count += 1
                elif caller == 'timer':
                    self.flush_by_timer_count += 1
                # Batch stats
                self.total_batches += 1
                self.total_batch_size += batch_len
                # Max and histogram
                if batch_len > self.max_batch_size:
                    self.max_batch_size = batch_len
                self.batch_size_counts[batch_len] = self.batch_size_counts.get(batch_len, 0) + 1

    def _flusher_loop(self):
        """Background thread that flushes small batches at a fixed interval to improve batching.
        """
        while not self._stop_event.is_set():
            time.sleep(self._flusher_interval)
            # Flush if any pending requests. Call _flush_batch() which will acquire the lock itself
            # Avoid holding the lock here to prevent deadlocks (we used to acquire the lock then call
            # _flush_batch which also tried to acquire the same lock).
            if self.current_batch:
                self._flush_batch(caller='timer')

    def get_flush_stats(self):
        """Return a tuple (size_flushes, timer_flushes)."""
        return (self.flush_by_size_count, self.flush_by_timer_count)

    def get_batch_stats(self):
        """Return (total_requests, total_batches, avg_batch_size).
        avg_batch_size is None when no batches have been sent yet.
        """
        avg = None
        if self.total_batches > 0:
            avg = float(self.total_batch_size) / float(self.total_batches)
        return (self.total_requests, self.total_batches, avg, self.max_batch_size, dict(self.batch_size_counts))


    def poll_predictions(self):
        """
        Procesa todo lo que haya llegado en output_queue
        y despacha a las colas correspondientes.
        """

        while True:
            results = self.output_queue.get(block=True)
            for tid, pred in results:
                if tid in self.pending:
                    self.pending[tid].put(pred)
                    del self.pending[tid]


        print("poll_predictions finalizado")

    def close(self):
        """
        Cierra el worker correctamente.
        """
        self._flush_batch()   # Asegura que las peticiones pendientes sean procesadas
        self.input_queue.put(None)  # Señal de cierre
        self.worker.join()  # Espera a que el worker finalice su ejecución
        # Stop flusher thread
        try:
            self._stop_event.set()
            if self._flusher_thread.is_alive():
                self._flusher_thread.join(timeout=1.0)
        except Exception:
            pass
        self.dispatch_loop_process.terminate()


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state  # Current board state
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this state
        self.children = []  # Child nodes (future possible states)
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Total reward (win/loss/draw) from simulations
        # Virtual loss for parallel MCTS and per-node lock
        self.virtual_loss = 0
        self.lock = threading.Lock()
        # Which player made the move that led to this node (True=White, False=Black), None for root
        self.player_just_moved = None

    def is_fully_expanded(self):
        # Returns True if all possible moves have been explored
        return len(self.children) == sum(1 for _ in self.state.legal_moves)


    def best_child(self, exploration_weight=1.4, virtual_loss_coef=0.0):
        # Select the child with the best value using UCT (Upper Confidence Bound for Trees)
        best_value = -float('inf')
        best_node = None
        for child in self.children:
            # Use virtual_loss to penalize children currently under exploration by other threads
            # Lower virtual_loss makes a child more attractive; subtract virtual_loss_coef * virtual_loss
            uct_value = (child.value / (child.visits + 1)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1)) - virtual_loss_coef * child.virtual_loss
            if uct_value > best_value:
                best_value = uct_value
                best_node = child
        return best_node
    
    def uct_children(self, exploration_weight=1.4, virtual_loss_coef=0.0):
        uct_values = []
        for child in self.children:
            # Use virtual_loss to penalize children currently under exploration by other threads
            # Lower virtual_loss makes a child more attractive; subtract virtual_loss_coef * virtual_loss
            uct_value = (child.value / (child.visits + 1)) + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1)) - virtual_loss_coef * child.virtual_loss
            uct_values.append((str(child.move),uct_value))
        #print(uct_values)
        return uct_values

        
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
    def __init__(self, root, get_best_function, chess_batcher=None, simulations=100, num_workers=None, virtual_loss_coef=1.0, virtual_loss_amount=1):
        """
        root: MCTSNode root
        get_best_function: callable(state) -> uci_move string
        chess_batcher: optional ChessBatcher instance used by get_best_function
        simulations: number of MCTS iterations
        """
        self.root = root  # Root node
        self.get_best_function = get_best_function  # Function to get the best move (callable)
        self.simulations = simulations  # Number of MCTS simulations
        # instancia de ChessBatcher para la inferencia vectorizada (opcional).
        self.chess_batcher = chess_batcher
        # If num_workers is not provided, set a higher default to increase concurrency
        if num_workers is None:
            try:
                cpu_count = mp.cpu_count()
            except Exception:
                cpu_count = 4
            # default to 2x CPUs but cap to a reasonable upper bound
            self.num_workers = min(64, max(4, cpu_count * 2))
        else:
            self.num_workers = num_workers
        # Virtual loss tuning
        self.virtual_loss_coef = virtual_loss_coef
        self.virtual_loss_amount = virtual_loss_amount

    def search(self, moves_top_k):
        # Usamos ThreadPoolExecutor porque la mayor parte del tiempo de MCTS 
        # (selección, expansión, backpropagation) es I/O-bound (esperando el Batcher) o CPU-bound ligero.
        # Run simulations in parallel to create concurrent model requests for batching.
        def worker_sim(_i):
            # Selection (with virtual loss) and expansion with minimal locking
            print("worker_sim",_i)
            node = self._select()
            if not node.is_fully_expanded():
                node = self._expand(node)
            # Simulation (may block on get_best and allow batching)
            winner, depth = self._simulate(node)
            # Backpropagate (will decrement virtual loss)
            self._backpropagate(node, winner)
            return depth

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as exe:
            # Submit all simulations; executor will schedule workers and allow many concurrent get_best calls
            futures = [exe.submit(worker_sim, i) for i in range(self.simulations)]
            # Collect depths for telemetry
            depths = []
            for f in concurrent.futures.as_completed(futures):
                try:
                    d = f.result()
                    if isinstance(d, int) or isinstance(d, float):
                        depths.append(d)
                except Exception:
                    pass

     
        ################### moves_top_k 
        children_uct_values = self.root.uct_children(exploration_weight=0)
        print(children_uct_values)
        cnn_prob_map = {move: prob for move, prob in moves_top_k}
        weighted_values = []
        for move, value in children_uct_values:
            # Buscar la probabilidad de la CNN. 
            # Usamos .get(move, 0.0) para que si el movimiento no está en cnn_probs,
            # se considere que tiene una probabilidad de 0, y el resultado sea 0.
            cnn_prob = cnn_prob_map.get(move, 0.0)
            
            # Calcular el nuevo valor ponderado
            weighted_value = value * cnn_prob
            
            # Añadir el resultado al nuevo array
            weighted_values.append((move, weighted_value))
        print(weighted_values)


        best = self.root.best_child(exploration_weight=0)
        # Print average depth telemetry if we collected any
        try:
            if 'depths' in locals() and depths:
                avg_depth = float(sum(depths)) / float(len(depths))
                print(f"MCTS simulation average depth: {avg_depth} (n={len(depths)})")
        except Exception:
            pass

        return best.move if best is not None else None, self.root.to_json()

    def _select(self):
        # Traverse down the tree to find a leaf node starting from root.
        # We implement selection with virtual loss: when we descend to a child we increment its virtual_loss
        # so other threads tend to avoid the same child.
        cur = self.root
        while cur.is_fully_expanded():
            # choose best child considering virtual loss
            child = cur.best_child(exploration_weight=1.4, virtual_loss_coef=self.virtual_loss_coef)
            if child is None:
                break
            # increment virtual loss for chosen child under its lock
            with child.lock:
                child.virtual_loss += self.virtual_loss_amount
            cur = child
        return cur

    def _expand(self, node):
        # Expand one of the children (possible moves from the current position)
        # Expand one move atomically to avoid races when multiple threads expand the same node
        legal_moves = list(node.state.legal_moves)
  


        with node.lock:
            for move in legal_moves:
                # Check if this state is already explored
                if not any(child.move == move for child in node.children):
                    new_state = node.state.copy()  # Clone the board to simulate the move
                    new_state.push(move)  # Apply the move
                    new_node = MCTSNode(new_state, parent=node, move=move)
                    # The player who just moved to reach new_state is the player who was to move at the parent
                    try:
                        new_node.player_just_moved = node.state.turn
                    except Exception:
                        new_node.player_just_moved = None
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
        """
        Perform a playout using get_best to simulate moves.
        Returns a tuple (winner_str, depth) where winner_str is one of '1-0','0-1','1/2-1/2'.
        If the playout exceeds the depth cap (60), evaluate the position with _simple_eval
        and return a winner based on material advantage.
        """
        DEPTH_CAP = 60
        state = node.state.copy()
        depth = 0
        while not state.is_checkmate() and not state.is_game_over() and sum(1 for _ in state.legal_moves) > 0:
            # Ask the supplied predictor for a move (string expected)
            best_move_str = self.get_best_function(state)

            # Prevent overly long playouts
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

                return result, depth

            # Validate and convert into a chess.Move object; fall back to a random legal move on any error
            try:
                if not isinstance(best_move_str, str):
                    raise ValueError("predicted move is not a UCI string")
                uci_move = chess.Move.from_uci(best_move_str)
                if uci_move not in state.legal_moves:
                    raise ValueError("predicted move not legal in this state")
            except Exception:
                # Choose a random legal move as fallback
                uci_move = random.choice(list(state.legal_moves))

            # Apply the Move object
            state.push(uci_move)
            depth += 1

        # If game finished normally, return its result and depth
        # state.result() returns strings like '1-0','0-1','1/2-1/2'
        return state.result(), depth

    def _backpropagate(self, node, winner):
        """Propaga el resultado hacia arriba, ajustando el signo según quién mueve."""
        while node is not None:
            node.visits += 1

            # Resultado desde la perspectiva de blancas
            if winner == '1-0':
                result = 1.0
            elif winner == '0-1':
                result = -1.0
            else:
                result = 0.5

            # Si el nodo representa el turno de negras, invertir el signo
            if not node.state.turn:  # False = negras
                result = -result

            node.value += result

            # Decrementar virtual loss si procede
            with node.lock:
                if hasattr(self, 'virtual_loss_amount') and node.virtual_loss > 0:
                    node.virtual_loss = max(0, node.virtual_loss - self.virtual_loss_amount)

            node = node.parent


# Example usage:
# Assuming we have a chess board state (from the `chess` library) and the `get_best` function
#def get_best_move(board):
    # Your CNN function that predicts the best move for the given board
#    return predict_chess_move(board.fen(),model, device)
def chessmarro_predict_top_k_moves(fen, model, device, k=5):
    """
    Realiza una única inferencia vectorial con el modelo de ajedrez para predecir
    los k movimientos con mayor probabilidad según la red neuronal.

    Parámetros:
    - fen (str): La notación FEN de la posición actual.
    - model (nn.Module): El modelo de red neuronal cargado.
    - device (torch.device/str): El dispositivo donde ejecutar la inferencia ('cuda' o 'cpu').
    - k (int): El número de movimientos principales (top-k) a devolver.
    
    Retorna:
    - list[tuple(str, float)]: Una lista de tuplas (movimiento_uci, probabilidad)
      ordenada de mayor a menor probabilidad.
    """
    # 1. Preparar el tensor del tablero
    # La función asume que concat_fen_legal devuelve un tensor de PyTorch (o algo convertible)
    # y ya maneja la legalidad del movimiento.
    board_tensor = concat_fen_legal(fen)
    if not isinstance(board_tensor, torch.Tensor):
        # Convertir a tensor de float32, la forma esperada por el modelo
        board_tensor = torch.tensor(board_tensor, dtype=torch.float32)

    # Añadir una dimensión de batch (de [C, H, W] a [1, C, H, W])
    boards_tensor = board_tensor.unsqueeze(0)
    
    # Mover al dispositivo (CPU/GPU)
    boards_tensor = boards_tensor.to(device)
    
    # 2. Realizar la predicción de probabilidades
    # Usaremos una versión simplificada de predict_chess_moves_vectorized para obtener las probabilidades
    # en lugar de muestrear un único movimiento.

    B = boards_tensor.size(0) # B debe ser 1 en este caso

    with torch.no_grad():
        logits = model(boards_tensor)  # [1, 4096]

    # --- LEGAL MASK ---
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).bool()

    # --- SANITIZE LOGITS (Reutilizando la lógica de predict_chess_moves_vectorized) ---
    finfo = torch.finfo(logits.dtype)
    negbig = finfo.min / 4
    logits = torch.nan_to_num(logits, nan=0.0, posinf=finfo.max/4, neginf=finfo.min/4)

    # --- MASK: Set illegal moves to a very low value ---
    masked = logits.masked_fill(~legal_masks, negbig)

    # --- SOFTMAX ---
    # Usamos temperatura T=1.0 para obtener la probabilidad "cruda" del modelo.
    temperature = 1.0 
    probs = torch.softmax(masked / temperature, dim=1) # [1, 4096]

    # --- HANDLE INVALID ROWS (Normalización) ---
    probs = torch.clamp(probs, min=0.0)
    row_sums = probs.sum(dim=1, keepdim=True)
    valid = (row_sums > 0) & torch.isfinite(row_sums)
    
    # Si la fila es válida, normalizar; si no, usar distribución uniforme legal como fallback.
    # En un escenario normal de juego, row_sums será ~1.0.
    if valid.item():
        final_probs = probs / row_sums
    else:
        # Fallback a distribución uniforme sobre legales (debería ser raro)
        uniform = legal_masks.float()
        final_probs = uniform / uniform.sum(dim=1, keepdim=True).clamp(min=1)
    
    # Asegurarse de que estamos trabajando con el tensor 1D de probabilidades para la única posición.
    final_probs = final_probs.squeeze(0) # [4096]

    # 3. Obtener el Top-K de movimientos
    # Obtener los k índices con la mayor probabilidad.
    top_k_values, top_k_indices = torch.topk(final_probs, k=k)

    # 4. Convertir índices a movimientos UCI y emparejar con la probabilidad
    top_moves_data = []
    # top_k_indices y top_k_values son tensores en el dispositivo, mover a CPU para procesamiento.
    top_k_indices_cpu = top_k_indices.cpu().numpy()
    top_k_values_cpu = top_k_values.cpu().numpy()

    for idx, prob in zip(top_k_indices_cpu, top_k_values_cpu):
        move_uci = number_to_uci(int(idx))
        top_moves_data.append((move_uci, float(prob)))

    print(f"Top {k} moves predicted: {top_moves_data}")
    return top_moves_data


def chessmarro_mcts_predict_chess_move(fen, simulations, model, device, batch_size=32, flusher_interval=0.1, num_workers=20, virtual_loss_coef=1.0, virtual_loss_amount=1):
    # Set up the initial chess board state
    
    # Prejuicios por IA para reforzar el caracter del despligue y necesitar menos simulaciones
    moves_top_k = chessmarro_predict_top_k_moves(fen,model,device)
    
    board = chess.Board(fen)

    # Create the root node
    root = MCTSNode(state=board)

    # Create a manager and a ChessBatcher to perform vectorized predictions
    manager = Manager()
    # Use provided batch_size (can be tuned)
    batcher = ChessBatcher(batch_size, model, device, manager=manager, flusher_interval=flusher_interval)

    # Define a get_best function that enqueues the state and waits for the batched prediction
    def get_best(state):
        # Convert state to fen and board tensor
        fen_local = state.fen()
        board_tensor = concat_fen_legal(fen_local)
        # Ensure it's a torch tensor of floats on CPU for batching; worker will move to device
        if not isinstance(board_tensor, torch.Tensor):
            board_tensor = torch.tensor(board_tensor, dtype=torch.float32)
        board_tensor = board_tensor.to(torch.float32)

        # Create a response queue and submit to batcher
        response_q = manager.Queue()
        batcher.add_board(board_tensor, response_q=response_q)
        # Do NOT force a flush here; background flusher will coalesce requests into batches.

        # Wait for the prediction (blocking)
        try:
            pred = response_q.get()
        except Exception:
            # On failure, fall back to direct prediction
            print("# On failure, fall back to direct prediction")
            pred = predict_chess_move(fen_local, model, device)
        return pred


    # Initialize MCTS with the wrapped get_best function
    mcts = MCTS(root=root, get_best_function=get_best, chess_batcher=batcher, simulations=simulations, num_workers=num_workers, virtual_loss_coef=virtual_loss_coef, virtual_loss_amount=virtual_loss_amount)

    # Run the MCTS algorithm and get the best move
    best_move, json = mcts.search(moves_top_k)

    # Optionally report flush stats before closing
    try:
        size_flushes, timer_flushes = batcher.get_flush_stats()
        print(f"Batcher flushes — by size: {size_flushes}, by timer: {timer_flushes}")
    except Exception:
        pass

    try:
        total_requests, total_batches, avg_batch, max_batch, batch_hist = batcher.get_batch_stats()
        print(f"Batch stats — requests: {total_requests}, batches: {total_batches}, avg batch size: {avg_batch}, max batch: {max_batch}")
        # Print small histogram (only sizes with at least one occurrence)
        if batch_hist:
            print("Batch size distribution (size:count):", sorted(batch_hist.items()))
    except Exception:
        pass

    # Close batcher/worker processes
    try:
        batcher.close()
    except Exception:
        pass

    # Display the best move
    print(f"The best move predicted is: {best_move}")
    return best_move, json


