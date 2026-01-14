# main.py
import multiprocessing as mp
import uvicorn
from api import create_api
from inference_server import inference_server
from chessgamemultithread import ChessBatcher
#from mcts_worker import mcts_worker
#from gradio_app import launch_gradio


# -------- Modelo lazy con lock --------
_model = None
_device = None
_model_lock = Lock()

def get_model():
    global _model, _device
    if _model is None:
        with _model_lock:
            if _model is None:
                print("Inicializando modelo (solo una vez en este proceso)...")
                try:
                    _model, _device = init_model()
                except Exception as e:
                    print(f"Warning: no se pudo inicializar el modelo: {e}")
                    _model, _device = None, None
    return _model, _device

# MCTS worker function. Esta función llama al mcts, el cual a su vez llama al servidor de inferencia.
# El servidor de inferencia está en otro proceso y se comunica mediante colas.
# El servidor de inferencia utiliza la GPU para hacer las predicciones.
# El MCTS también crea procesos para paralelizar las simulaciones.
# Cada proceso de MCTS envía solicitudes de inferencia al servidor de inferencia y espera las respuestas.
# Finalmente, el MCTS devuelve el mejor movimiento encontrado.
# MCTS necesita también de un proceso que escuche peticiones de inferencia de cualquier worker o simulación interna del worker.
# Ese proceso batcher une las peticiones en batches y las envía al servidor de inferencia.
def mcts_worker(inference_q, inference_response_q, mcts_result_q):
    id = mp.current_process().pid
    print("MCTS worker started", id)
    for _ in range(10):
        inference_q.put(("_task"+str(_), None))
        response = inference_response_q.get()
        #print(f"MCTS worker {id} received response: {response}")
    mcts_result_q.put((id, f"move_from_worker_{id}"))
    print("MCTS worker finished", id)

def task_listener(task_q, mcts_result_q, inference_q, inference_response_q, tasks_result_q):
    print("Task listener started")
    while True:
        task = task_q.get()
        print("Task listener received task:", task)
        workers = [
            mp.Process(
                target=mcts_worker,
                args=(inference_q, inference_response_q, mcts_result_q),
            )
            for _ in range(mp.cpu_count())
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        print("All MCTS workers finished for task:", task, mcts_result_q)
        # Mock result after all workers are done
        results = [mcts_result_q.get() for _ in range(mp.cpu_count())]
        tasks_result_q.put((task[0], results))  # Just a placeholder
    print("Task listener finished")    

def main():
    print("Main process started")

    inference_q = mp.Queue()   # Queue for inference requests
    inference_response_q = mp.Queue() # Queue for inference responses
    task_q = mp.Queue()    # Queue for MCTS tasks
    mcts_result_q = mp.Queue()  # Queue for MCTS results
    tasks_result_q = mp.Queue() # Queue for final task results

    # Start the inference server process
    gpu = mp.Process(
        target=inference_server,
        args=(inference_q, inference_response_q),
    )
    gpu.start()

    # Start the task listener process
    task_listener_process = mp.Process(
        target=task_listener,
        args=(task_q, mcts_result_q, inference_q, inference_response_q, tasks_result_q),
    )
    task_listener_process.start()

    # Start the FastAPI server
    api_app = create_api(task_q, tasks_result_q)
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

    # Start the batcher process for MCTS inference requests
    batcher = ChessBatcher(64, model, device, manager=manager, flusher_interval=flusher_interval)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    get_model()
    main()