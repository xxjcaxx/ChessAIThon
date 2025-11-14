# Documentación técnica: despliegue y MCTS batched (Gradio + FastAPI)

Fecha: 2025-11-14

Este documento describe los ficheros principales del despliegue del modelo MCTS batching en este repositorio:

- `modelDeploy/app.py` — interfaz web (Gradio) + API REST (FastAPI) y carga perezosa del modelo.
- `modelDeploy/chessgamemultithread.py` — implementación del MCTS, batching de inferencia y utilidades asociadas.
- `modelDeploy/predict.py` — script CLI que inicializa el modelo y lanza una predicción (ejemplo de uso fuera del servidor web).

## Objetivo

Permitir inferencia eficiente de jugadas con un modelo CNN que predice distribuciones sobre 4096 movimientos (mapa 64x64), agrupando (batching) las peticiones desde múltiples hilos/procesos hacia una GPU para mejorar throughput.

---

## Resumen de alto nivel

1. `app.py` expone dos interfaces: una UI con Gradio (por defecto en `0.0.0.0:7860`) y un endpoint REST minimalista con FastAPI (`0.0.0.0:8000/predict`).
2. `chessgamemultithread.py` implementa:
   - `ChessBatcher`: cola y batching para enviar tensores al worker GPU.
   - `batch_predict_worker`: proceso que recibe batches, ejecuta el modelo y devuelve predicciones UCI.
   - `MCTS` y `MCTSNode`: algoritmo de búsqueda Monte Carlo Tree Search multihilo que usa el `ChessBatcher` para pedir movimientos durante las simulaciones.
3. `predict.py` es un wrapper/runner CLI que carga el modelo y invoca `chessmarro_mcts_predict_chess_move` (similar a la llamada interna de `predict_fn`).

---

## Fichero: `chessgamemultithread.py`

### Componentes clave

- predict_chess_moves_vectorized(boards_tensor, temperature, model)
  - Recibe un tensor con B posiciones en batch.
  - El modelo devuelve logits [B, 4096].
  - Aplica máscara de legales (extraída de `boards_tensor`), sanitiza logits, softmax con temperatura y normaliza filas inválidas.
  - Devuelve una lista de movidas en formato UCI usando `number_to_uci`.

- batch_predict_worker(input_queue, output_queue, model, device)
  - Proceso independiente que continuamente lee `input_queue`, apila tensores, mueve al `device` y llama a `predict_chess_moves_vectorized`.
  - Envia [(task_id, uci_move), ...] a `output_queue`.

- dispatch_loop(output_queue, pending)
  - Proceso que toma resultados de `output_queue` y los despacha a las colas `pending[task_id]` de cada cliente.

- ChessBatcher
  - Acumula peticiones en `current_batch` y las manda al worker cuando se alcanza `batch_size` o cuando un flusher tiempo las descarga.
  - Cada `add_board` crea un `task_id` y devuelve una `Queue` de respuesta (usando `Manager`) donde el cliente esperará la predicción.
  - Mantiene estadísticas de batches, flusher por temporizador y por tamaño.

- MCTSNode
  - Campos principales: `state` (objeto chess.Board), `move`, `children`, `visits`, `value`, `virtual_loss`, `lock`, `player_just_moved`.
  - `to_dict()` y `to_json()` para serializar (usa `str(self.move)` para representar la jugada).

- MCTS
  - Entrada: `root` (MCTSNode), `get_best_function` (callable que recibe un `state` y devuelve UCI string), y opciones de concurrencia.
  - Flujo: `search()` lanza N simulaciones usando `ThreadPoolExecutor`.
  - `_select()`: desciende por UCT + virtual loss, marcando virtual loss en hijos seleccionados.
  - `_expand()`: crea un nuevo `MCTSNode` por una jugada legal (sin evaluación IA en expansión en la versión actual).
  - `_simulate()`: ejecuta un playout donde en cada paso llama a `get_best_function(state)`; espera un string UCI (si falla, escoge un legal aleatorio).
  - `_backpropagate()`: actualiza `visits` y `value` propagando el resultado (invirtiendo signo según quién tenga el turno).

- chessmarro_mcts_predict_chess_move(fen, simulations, model, device, ...)
  - Inicializa `Board` y `MCTSNode` root.
  - Crea `Manager` + `ChessBatcher` y define `get_best(state)` que convierte `state` en tensor (`concat_fen_legal`), manda la petición al `batcher` y espera la respuesta.
  - Lanza `MCTS` con `get_best` y devuelve `(best_move, tree_json)`.

### Notas de implementación y rendimiento

- El worker usa mixed-precision en CUDA con `torch.cuda.amp.autocast()` cuando hay GPU disponible.
- El uso de `Manager().Queue()` y `Process` permite aislar la inferencia en procesos separados y evitar saturar la CPU principal.
- Se usan colas para mapear `task_id` a respuesta, evitando bloqueos y permitiendo múltiples clientes.

---

## Fichero: `app.py`

### Arquitectura

- Carga perezosa del modelo: `get_model()` realiza `init_model()` dentro de un lock global. Esto evita que la inicialización (y sus logs) se repitan si el módulo se importa varias veces o si hay procesos adicionales.
- `predict_fn(fen, simulations)`: wrapper que Gradio llama; llama a `get_model()` y después a `chessmarro_mcts_predict_chess_move`.
- Interfaz Gradio: definida con `gr.Interface(...)`. Por conveniencia se lanza con `server_name='0.0.0.0'` y `server_port=7860`.
- FastAPI: aplica un endpoint POST `/predict` que llama a `predict_fn` y siempre devuelve `{"move": "<uci>"}` (se fuerza `str(move)` para asegurar consistencia entre Gradio y FastAPI).
- Lanzamiento combinado (cuando `__main__`):
  - Establece `mp.set_start_method('spawn')` (necesario para PyTorch + CUDA en Unix).
  - Lanza Gradio en un hilo daemon y luego UVicorn para FastAPI en el hilo principal (puerto 8000).

### Endpoints expuestos

- UI Gradio: http://<host>:7860/ (por defecto `0.0.0.0:7860`).
- API REST (FastAPI): POST http://<host>:8000/predict
  - Payload JSON: `{ "fen": "<FEN>", "simulations": <N> }`
  - Respuesta JSON: `{ "move": "e2e4" }`

---

## Fichero: `predict.py` (script CLI)

- Inicializa multiprocessing con `mp.set_start_method('spawn')`.
- Llama a `init_model()` para cargar el modelo y el dispositivo.
- Ejecuta `chessmarro_mcts_predict_chess_move(fen, simulations, model, device, num_workers=64)` y muestra el movimiento resultante.

---

## Ejemplos prácticos (curl)

- Llamada FastAPI (JSON simple):

```bash
curl -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/predict -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","simulations":10}'
```

- Llamada Gradio (endpoint interno de Gradio API):

```bash
curl -H "Content-Type: application/json" -X POST http://127.0.0.1:7860/api/predict/ -d '{"data":["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",10],"fn_index":0}'
```

- Ejecutar script directamente (sin servidor):

```bash
python modelDeploy/predict.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 10
```

---

## Concurrencia, procesos y consejos de depuración

1. Multiprocessing start method
   - Para PyTorch+CUDA, `spawn` es recomendado en Unix para evitar problemas de fork con CUDA context.
   - Fija `mp.set_start_method('spawn', force=True)` solo en `if __name__ == '__main__'` para evitar efectos secundarios al importar el módulo.

2. Evitar doble carga del modelo
   - `get_model()` implementa carga perezosa con un `Lock` por proceso; si aún ves mensajes duplicados puede deberse a que hay varios procesos (cada proceso carga su propio modelo) creados por `Process` o por Gradio/uvicorn workers.

3. Bloqueos y virtual loss
   - `MCTS` usa `virtual_loss` para evitar que varios hilos exploren el mismo hijo al mismo tiempo. Cada `MCTSNode` tiene su propio `lock` para modificaciones atómicas.

4. Depuración de batching
   - Imprime `batch_hist` y estadísticas de `ChessBatcher.get_batch_stats()` para ver si el flusher está coalesciendo correctamente.

---

## Recomendaciones y mejoras (rápidas)

- Salida consistente: ya se fuerza `str(move)` al devolver resultados para evitar distintos formatos entre Gradio y FastAPI.
- Evaluación en expansión: si quieres inicializar nodos hijos con una valoración de la IA (prior o value), modifica `_expand()` para llamar al modelo por cada hijo (o mejor: vectoriza/agrupa las evaluaciones de hijos antes de crear los nodos) y asigna `node.value`, `node.visits` (pseudo-visitas) y `node.total_value` o `node.prior` según tu esquema (esto requiere cambiar la clase `MCTSNode`).
- Tests: Añadir tests unitarios rápidos para `ChessBatcher` (simular requests y comprobar que se reciben respuestas) y para la función `predict_chess_moves_vectorized` con tensores sintéticos.
- Seguridad: si `share=True` está activo en Gradio, recuerda que crea una URL pública; desactívala en entornos de producción.

---

## Preguntas frecuentes rápidas

- ¿Por qué Gradio y FastAPI devolvían formatos distintos?  
  Porque Gradio mostraba el string directamente en la UI; FastAPI serializaba el valor retornado por `predict_fn` (si es un `chess.Move` u otro objeto, JSON resultante puede ser diferente). Por eso en `app.py` se normalizó la salida a `str(move)`.

- ¿Cómo evitar que el modelo se cargue dos veces?  
  Carga perezosa por proceso (`get_model()` + `Lock`) y evitar `set_start_method` en tiempo de import. Aun así, cada proceso hijo que use GPU deberá cargar su propio modelo.

---

## Próximos pasos sugeridos

- Implementar la inicialización de hijos en `_expand()` basada en la evaluación de la IA si quieres mejorar la calidad de PUCT/PUCT-like (este cambio requiere diseñar cómo normalizar priors y actualizar `MCTSNode`).
- Añadir tests de integración que ejecuten `chessmarro_mcts_predict_chess_move` con un modelo dummy (por ejemplo, un modelo que devuelve logits uniformes) para validar la pipeline sin GPU.


---

Archivo generado automáticamente por la herramienta de documentación del proyecto.
