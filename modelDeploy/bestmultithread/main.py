# main.py
import multiprocessing as mp
from threading import Lock
import threading
import uvicorn
from .api import create_api
from .inference_server import inference_server
from chessmodel import init_model
from .batcher import batcher_loop
from .mcts import mcts_worker
import queue
from collections import Counter
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




def task_listener(task_q, mcts_result_q, batcher_q, tasks_result_q, worker_response_queues, last_batch_avg):
    print("Task listener started")
    n_workers = mp.cpu_count()
    while True:
        task = task_q.get()
        print("Task listener received task:", task)
        workers = [
            mp.Process(
                target=mcts_worker,
                args=(batcher_q, mcts_result_q, worker_response_queues[i], i, task, 0.5 + (i * (1.89 / (n_workers - 1)))),
            )
            for i in range(n_workers)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        print("All MCTS workers finished for task:", task)
        # Mock result after all workers are done
        results = [mcts_result_q.get() for _ in range(n_workers)]
        #print("MCTS results collected:", results)
        # 1. Acumular todas las visitas en un contador global
        total_visits = Counter()

        for id_worker, best_move_worker, move_counts in results:
            # move_counts es la lista de tuplas [('f7f5', 3), ('c8h3', 1), ...]
            for move, visits in move_counts:
                total_visits[move] += visits

        # 2. Determinar la jugada ganadora (la que tiene más visitas totales)
        if total_visits:
            best_move_final = max(total_visits, key=total_visits.get)
            total_score = total_visits[best_move_final]
            
            print(f"Mejor jugada final: {best_move_final} con {total_score} visitas totales")
        else:
            best_move_final = None
        
        # imprimir las visitas de la mejor jugada en cada worker
        for id_worker, best_move_worker, move_counts in sorted(results, key=lambda x: x[0]):
            # Buscar el move_counts específico para la mejor jugada sugerida por este worker
            move_visits_from_worker = next((visits for move, visits in move_counts if move == best_move_worker), 0)
            print(f"Worker {id_worker} sugirió {best_move_worker} con {move_visits_from_worker} visitas en su búsqueda")


        tasks_result_q.put((task[0], best_move_final, total_visits)) 
        print("Total visits:", total_visits)
        avg, total, todas = last_batch_avg[0], last_batch_avg[1], last_batch_avg[2]
        print(f"Media: {avg}, Incompletos: {total}, Todos los batches: {todas}")
    print("Task listener finished")    

def main():
    print("Main process started")

    inference_q = mp.Queue()   # Queue for inference requests
    inference_response_q = mp.Queue() # Queue for inference responses
    task_q = mp.Queue()    # Queue for MCTS tasks
    mcts_result_q = mp.Queue()  # Queue for MCTS results
    tasks_result_q = mp.Queue() # Queue for final task results
    batcher_q = mp.Queue()  # Queue for batcher requests
    worker_response_queues = [mp.Queue(maxsize=128) for _ in range(mp.cpu_count())]
    last_batch_avg = mp.Array('d', [0.0, 0.0, 0.0])

    # Start the inference server process
    gpu = mp.Process(
        target=inference_server,
        args=(inference_q, inference_response_q, _model, _device),
    )
    gpu.start()

    # Start the task listener process
    task_listener_process = mp.Process(
        target=task_listener,
        args=(task_q, mcts_result_q, batcher_q, tasks_result_q, worker_response_queues, last_batch_avg),
    )
    task_listener_process.start()

    # Start the batcher process
    batcher = mp.Process(target=batcher_loop, args=(batcher_q, worker_response_queues, inference_q, inference_response_q, last_batch_avg))
    batcher.start()

    # Start the FastAPI server
    api_app = create_api(task_q, tasks_result_q)
    uvicorn.run(api_app, host="0.0.0.0", port=8000)






def run():
    print("Starting main process...", __name__)
    get_model()
    main()