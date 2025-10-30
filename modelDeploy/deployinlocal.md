python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py 



To deploy we have to create a docker container with dockerfile and docker-compose Docker should be able to use GPU

Asegúrate de que nvidia-smi funcione correctamente fuera de Docker.

sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

Para que funcione bien torch dentro del docker hay que hacer un dockerfile con una base de FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime  

¡Es un gran objetivo\! El MCTS es un algoritmo que se beneficia enormemente del paralelismo, ya que cada simulación puede considerarse una tarea independiente.

Tu implementación actual de MCTS es clásica. Para adaptarla al multiprocesamiento y explorar diferentes ramas a la vez, el enfoque más limpio y moderno es usar tu clase **`ChessBatcher`** junto con **`multiprocessing.Pool`** o un **`concurrent.futures.ThreadPoolExecutor`** (más adecuado para tareas ligadas a la CPU como la búsqueda) para las fases de Selección y Backpropagation, mientras que la fase de Simulación (la inferencia del modelo) se delega a tu *worker* de GPU.

-----

## 🏗️ Adaptación del MCTS a Multiproceso (MCTS Batching)

La clave es modificar el método `MCTS.search()` para que ejecute múltiples ciclos de MCTS en paralelo (o al menos hasta la fase de simulación) y agrupe las peticiones de inferencia para tu `ChessBatcher`.

### 1\. La Función `_simulate` debe cambiar a Inferencia Batch

El método `_simulate` ya no puede ejecutar la simulación completa de la partida. En su lugar, debe hacer una **única petición** de inferencia al modelo a través de tu *Batcher* y devolver la cola de respuesta.

### 2\. La Función `search` debe gestionar el Paralelismo y el Batching

Necesitas un *pool* de procesos o hilos para manejar muchas búsquedas concurrentes y un sistema para enviar los estados de los nodos a tu `ChessBatcher`.

Aquí tienes un esqueleto de cómo adaptarías tu clase **`MCTS`** y la función **`_simulate`** para el paralelismo:

```python
import concurrent.futures
import multiprocessing as mp
import random
import chess
# Asumimos que MCTSNode y number_to_uci están definidos en otro lugar
# Asumimos que ChessBatcher y las funciones de Queue están accesibles

class MCTS:
    def __init__(self, root, chess_batcher, simulations=100, num_workers=4):
        self.root = root
        # Ahora recibe una instancia de ChessBatcher para la inferencia vectorizada.
        self.chess_batcher = chess_batcher 
        self.simulations = simulations
        self.num_workers = num_workers # Número de hilos/procesos para MCTS

    def search(self):
        # Usamos ThreadPoolExecutor porque la mayor parte del tiempo de MCTS 
        # (selección, expansión, backpropagation) es I/O-bound (esperando el Batcher) o CPU-bound ligero.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            
            # Un mapa para rastrear los resultados de las simulaciones pendientes
            futures = []
            
            for i in range(self.simulations):
                # Usamos submit para iniciar un ciclo MCTS en un hilo/worker diferente.
                future = executor.submit(self._run_one_simulation)
                futures.append(future)

            # Esperamos a que todas las simulaciones terminen y procesamos los resultados.
            # Esta parte podría ser más eficiente, pero es simple y correcta.
            for future in concurrent.futures.as_completed(futures):
                try:
                    node, winner = future.result()
                    self._backpropagate(node, winner)
                except Exception as e:
                    print(f"Error en la simulación: {e}")

        # Después de todas las simulaciones, devuelve el mejor movimiento.
        return self.root.best_child(exploration_weight=0).move
    
    # -----------------------------------------------------
    # Funciones de Soporte
    # -----------------------------------------------------

    def _run_one_simulation(self):
        """Ejecuta una simulación completa de MCTS (Selección -> Expansión -> Simulación)."""
        
        # Paso 1: Selección (Se ejecuta en el hilo/worker)
        node = self._select(self.root) 

        # Paso 2: Expansión (Se ejecuta en el hilo/worker)
        if not node.is_fully_expanded():
            node = self._expand(node)
        
        # Paso 3: Simulación (Petición Asíncrona a la GPU)
        # Aquí es donde delegamos la inferencia del modelo al ChessBatcher.
        # En el MCTS tradicional, se haría un playout aleatorio.
        # En el MCTS guiado por CNN, solo se necesita un movimiento del modelo.
        
        # El batcher necesita el tensor de entrada (el estado del tablero)
        # ⚠️ IMPORTANTE: Necesitas una función para convertir el objeto 'state' 
        # a un tensor que incluya la máscara legal (como boards_tensor en predict_chess_moves_vectorized).
        board_tensor_for_cnn = self._state_to_tensor(node.state) 

        # Añade la petición al batcher, que devolverá la cola donde estará el resultado.
        response_queue = self.chess_batcher.add_board(board_tensor_for_cnn)
        
        # Bloquea el hilo/worker de MCTS hasta que la GPU devuelva el resultado.
        predicted_move_uci = response_queue.get(block=True) 

        # El resto de la simulación sigue el antiguo _simulate, pero ahora es una "simulación" de 1 paso.
        state = node.state.copy()
        
        # 3b: Aplicar el movimiento predicho por la CNN.
        if chess.Move.from_uci(predicted_move_uci) not in state.legal_moves:
            # Si la predicción es ilegal (lo que no debería pasar si se usa legal_masks), elige una legal.
            predicted_move_uci = random.choice(list(state.legal_moves)).uci()
            
        state.push_uci(predicted_move_uci)
        
        # Simplificamos el _simulate a solo la predicción de 1 paso y asumimos un resultado simplificado
        # o se continúa con un playout aleatorio (como tu original).
        # Para un MCTS con Red Neuronal (AlphaZero/Stockfish), se suele usar la "Policy" (probabilidades) 
        # y el "Value" (valor de la posición) del modelo para guiar la búsqueda y terminar aquí.
        
        # Para mantener la estructura original (aunque menos eficiente):
        winner = self._continue_playout(state)
        
        # Devuelve el nodo y el resultado para la Backpropagation.
        return node, winner

    # -----------------------------------------------------
    # Métodos Auxiliares
    # -----------------------------------------------------

    def _select(self, node):
        # Lógica original de selección UCT
        while node.is_fully_expanded():
            node = node.best_child() 
        return node

    def _expand(self, node):
        # Lógica original de expansión
        # ... (Mantienes tu código original de _expand)
        legal_moves = node.state.legal_moves
        for move in legal_moves:
            # ... (código de creación de nuevo nodo)
            if not any(child.move == move for child in node.children):
                new_state = node.state.copy()
                new_state.push(move)
                new_node = MCTSNode(new_state, parent=node, move=move)
                node.children.append(new_node)
                return new_node
        return node

    def _continue_playout(self, state):
        """Simulación de la partida hasta el final (playout) - para mantener tu lógica original."""
        while not state.is_checkmate() and not state.is_game_over() and sum(1 for _ in state.legal_moves) > 0:
            # Aquí la predicción ya no debe ser CNN, sino un playout aleatorio.
            best_move = random.choice(list(state.legal_moves)).uci()
            state.push_uci(best_move)
        return state.result()

    def _backpropagate(self, node, winner):
        # Lógica original de backpropagation (debe ser thread-safe)
        # ... (Mantienes tu código original de _backpropagate. MCTSNode debe ser thread-safe o usar locks)
        while node is not None:
            node.visits += 1
            if winner == '1-0':
                node.value += 1
            elif winner == '0-1':
                node.value -= 1
            else:
                node.value += 0.5
            node = node.parent
            
    def _state_to_tensor(self, state):
        # ⚠️ ESTA FUNCIÓN ES CRUCIAL Y DEBE SER IMPLEMENTADA
        # Debe convertir el objeto chess.Board a un tensor de PyTorch 
        # con la codificación de canales que espera tu modelo Conv2D, 
        # incluyendo la máscara legal en los últimos 64 canales.
        raise NotImplementedError("Implementar la codificación de tablero a tensor, incluyendo la máscara legal.")
```

### 🔑 Puntos Clave del Paralelismo

1.  **Paralelismo de la Búsqueda (`ThreadPoolExecutor`):** Usar un `Executor` permite que varios hilos realicen los pasos de Selección y Expansión al mismo tiempo, explorando diferentes partes del árbol.
2.  **Delegación a GPU (`ChessBatcher`):** En lugar de que cada hilo ejecute la CNN individualmente (lo cual es lento), el MCTS delega la costosa fase de inferencia a la clase `ChessBatcher`.
3.  **Batching: La Ganancia de Velocidad:** Tu `ChessBatcher` agrupará las peticiones de inferencia de varios hilos de MCTS en un solo lote para la GPU.
4.  **`response_queue.get(block=True)`:** Esta es la parte de "sincronización". El hilo de MCTS se pausa hasta que el *worker* de la GPU termine de procesar el lote y devuelva el resultado de la inferencia.
5.  **`_backpropagate` Thread-Safe:** Debes asegurarte de que cuando varios hilos intenten actualizar las propiedades `visits` y `value` de un `MCTSNode` común, no haya condiciones de carrera. Esto a menudo requiere el uso de **`threading.Lock`** dentro de la clase `MCTSNode` o de `multiprocessing.Lock` si usas procesos.

Este patrón desacopla la lógica de la búsqueda (MCTS) de la lógica del hardware (GPU), permitiendo que la búsqueda se realice en paralelo de manera efectiva.