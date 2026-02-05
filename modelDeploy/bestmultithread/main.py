# main.py
import multiprocessing as mp
from threading import Lock
import threading
import uvicorn
from .api import create_api
from .inference_server import inference_server
from chessmodel import init_model
from .batcher import batcher_loop
from .mcts import  mcts_worker_persistent
import queue
from collections import Counter
import os
import psutil
import time
import chess
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
    n_workers = (mp.cpu_count() // 2 ) +  (mp.cpu_count() // 3 )
    print("\033[1;36m👷 Workers:\033[0m", n_workers)

    #print("Task listener started", n_workers, "MCTS workers will be launched")
    worker_task_in_queues = [mp.Queue(maxsize=1) for _ in range(n_workers)]
    # Lanzamos los workers UNA SOLA VEZ
    for i in range(n_workers):
        p = mp.Process(
            target=mcts_worker_persistent, # Nueva función
            args=(batcher_q, mcts_result_q, worker_response_queues[i], worker_task_in_queues[i], i, 0.5 + (i * (1.89 / (n_workers - 1))))
        )
        p.daemon = True
        p.start()

    while True:
        task = task_q.get()
        #print("Task listener received task:", task)
        board_demo = chess.Board(task[1])
        print(board_demo.unicode().replace("⭘", "·"))
        start = time.perf_counter()
        for q in worker_task_in_queues:
            q.put(task)

        results = [mcts_result_q.get() for _ in range(n_workers)]

        #print("All MCTS workers finished for task:", task)
        # Mock result after all workers are done
        #results = [mcts_result_q.get() for _ in range(n_workers)]
        #print("MCTS results collected:", results)
        # 1. Acumular todas las visitas en un contador global
        total_visits = Counter()

        for id_worker, best_move_worker, move_counts in results:
            # move_counts es la lista de tuplas [('f7f5', 3), ('c8h3', 1), ...]
            for move, visits in move_counts:
                total_visits[move] += visits

        # 2. Determinar la jugada ganadora (la que tiene más visitas totales)
        if total_visits:
            top_moves = total_visits.most_common(6)
            best_move_final = top_moves[0][0]  # El string de la jugada (ej: 'e2e4')
            total_score_final = top_moves[0][1] # Las visitas totales de esa jugada


            alternatives = top_moves[1:]
            #print(f"Alternativas encontradas: {len(alternatives)}")
            print(f"\033[1;32m🏆 Best move:\033[0m {best_move_final}  \033[1;33m({total_score_final} visits)\033[0m\n")
            #print(f"Mejor jugada final: {best_move_final} con {total_score_final} visitas totales")
        else:
            best_move_final = None
            total_score_final = 0
            alternatives = []
        
        # imprimir las visitas de la mejor jugada en cada worker
        # Formateamos los resultados de cada worker en una lista de strings cortos
        worker_info = [
            f"W{idw}:{mv}({next((v for m, v in cnts if m == mv), 0)})" 
            for idw, mv, cnts in sorted(results, key=lambda x: x[0])
        ]
        
        # Imprimimos todo en una sola línea separada por pipes
        print(f"-> Workers detail: {' | '.join(worker_info)}\n")


        tasks_result_q.put((task[0], best_move_final, total_score_final, alternatives)) 
        #print("Total visits:", total_visits)
        avg, total, todas = last_batch_avg[0], last_batch_avg[1], last_batch_avg[2]
        #print(f"Media: {avg}, Incompletos: {total}, Todos los batches: {todas}")
        print(
            f"\033[1;35m📊 Batcher:\033[0m "
            f"\033[34mavg={avg}\033[0m | "
            f"\033[33mincomplete={total}\033[0m | "
            f"\033[32mtotal={todas}\033[0m"
        )
        end = time.perf_counter()
        #print(f"Tiempo total para procesar la tarea: {end - start:.2f} segundos")
        print(f"\033[1;32m🚀 Processing time:\033[0m {end - start:.2f}  s\n")
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