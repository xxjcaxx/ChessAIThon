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




def task_listener(task_q, mcts_result_q, batcher_q, tasks_result_q, worker_response_queues):
    print("Task listener started")
    n_workers = 1 #mp.cpu_count()
    while True:
        task = task_q.get()
        print("Task listener received task:", task)
        workers = [
            mp.Process(
                target=mcts_worker,
                args=(batcher_q, mcts_result_q, worker_response_queues[i], i, task),
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
        tasks_result_q.put((task[0], results))  # Just a placeholder
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

    # Start the inference server process
    gpu = mp.Process(
        target=inference_server,
        args=(inference_q, inference_response_q, _model, _device),
    )
    gpu.start()

    # Start the task listener process
    task_listener_process = mp.Process(
        target=task_listener,
        args=(task_q, mcts_result_q, batcher_q, tasks_result_q, worker_response_queues),
    )
    task_listener_process.start()

    # Start the batcher process
    batcher = mp.Process(target=batcher_loop, args=(batcher_q, worker_response_queues, inference_q, inference_response_q))
    batcher.start()

    # Start the FastAPI server
    api_app = create_api(task_q, tasks_result_q)
    uvicorn.run(api_app, host="0.0.0.0", port=8000)




def run():
    print("Starting main process...", __name__)
    get_model()
    main()