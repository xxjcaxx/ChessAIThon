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
    """
    Predice movimientos para un batch de posiciones con Conv2D, usando
    movimientos legales codificados en los últimos 64 canales/elementos.
    """
    B = boards_tensor.size(0)
    device = boards_tensor.device

    with torch.no_grad(): # Ejecución del modelo. Conv2D espera [Batch, Canales, Alto, Ancho].
        outputs = model(boards_tensor)  # [B, 4096], conv2d espera [B, C, H, W]

    # Extraer máscaras legales de los últimos 64 canales
    # Flatten del canal y el tablero a 4096
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).to(dtype=torch.bool)

    # Sanitize outputs (replace nan/inf) using dtype-aware finite bounds to avoid overflow on float16
    finfo = torch.finfo(outputs.dtype)
    posinf = float(finfo.max) / 2.0
    neginf = float(finfo.min) / 2.0
    outputs = torch.nan_to_num(outputs, nan=0.0, posinf=posinf, neginf=neginf)

    # Use a large negative finite value instead of -inf to avoid producing NaNs in softmax
    NEG_INF = neginf
    device = outputs.device
    masked_logits = torch.where(legal_masks.to(device), outputs, torch.full_like(outputs, NEG_INF))

    # Softmax with temperature. If a row has no legal moves (all False), we'll handle it below.
    probs = torch.softmax(masked_logits / temperature, dim=1)  # [B, 4096]

    # Fix invalid rows: multinomial requires non-negative finite probabilities that sum > 0.
    probs = torch.clamp(probs, min=0.0)
    row_sums = probs.sum(dim=1)

    pred_moves = []
    for i in range(B):
        if not torch.isfinite(row_sums[i]) or row_sums[i] <= 0.0:
            # Fallback strategies in order:
            # 1) try to sample from outputs (ignoring mask) if they are finite
            out_row = outputs[i]
            # sanitize
            # Sanitize using dtype-aware bounds
            out_finfo = torch.finfo(out_row.dtype)
            out_posinf = float(out_finfo.max) / 2.0
            out_neginf = float(out_finfo.min) / 2.0
            out_row = torch.nan_to_num(out_row, nan=0.0, posinf=out_posinf, neginf=out_neginf)
            # If legal mask exists for this board, pick a random legal index
            if legal_masks[i].any():
                legal_idxs = torch.nonzero(legal_masks[i], as_tuple=True)[0]
                if legal_idxs.numel() > 0:
                    idx = int(legal_idxs[random.randrange(0, legal_idxs.numel())].item())
                    pred_moves.append(number_to_uci(idx))
                    continue

            # If all -INF-like, pick a random index
            if torch.all(out_row <= NEG_INF / 2):
                idx = random.randrange(0, out_row.numel())
            else:
                # take argmax of outputs as a deterministic fallback
                idx = int(torch.argmax(out_row).item())
            pred_moves.append(number_to_uci(idx))
        else:
            row_probs = probs[i]
            # normalize to sum 1 in case of rounding
            row_probs = row_probs / row_probs.sum()
            # multinomial expects 2D tensor
            try:
                sel = torch.multinomial(row_probs, num_samples=1).squeeze(0)
                idx = int(sel.item())
            except Exception:
                # As a last resort, pick argmax
                idx = int(torch.argmax(row_probs).item())
            pred_moves.append(number_to_uci(idx))

    return pred_moves

     
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

    def search(self):
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

        # Return the best move after simulations
        # Safe debug prints
       # print("Root children:", len(self.root.children))
       # try:
       #     print(self.root.to_json())
       # except Exception:
       #     pass

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
def get_best_move(board):
    # Your CNN function that predicts the best move for the given board
    return predict_chess_move(board.fen(),model, device)



def chessmarro_mcts_predict_chess_move(fen, simulations, model, device, batch_size=32, flusher_interval=0.1, num_workers=20, virtual_loss_coef=1.0, virtual_loss_amount=1):
    # Set up the initial chess board state
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
            pred = predict_chess_move(fen_local, model, device)
        return pred

    # Initialize MCTS with the wrapped get_best function
    mcts = MCTS(root=root, get_best_function=get_best, chess_batcher=batcher, simulations=simulations, num_workers=num_workers, virtual_loss_coef=virtual_loss_coef, virtual_loss_amount=virtual_loss_amount)

    # Run the MCTS algorithm and get the best move
    best_move, json = mcts.search()

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