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
import subprocess
from pathlib import Path
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
    n_workers = 1 # ((mp.cpu_count() // 2 ) +  (mp.cpu_count() // 4 ))//4
    print("\033[1;36m👷 Workers:\033[0m", n_workers)

    #print("Task listener started", n_workers, "MCTS workers will be launched")
    worker_task_in_queues = [mp.Queue(maxsize=1) for _ in range(n_workers)]
    # Lanzamos los workers UNA SOLA VEZ
    for i in range(n_workers):
        # Distribuir puct uniformemente entre 0.5 y 1.4 independientemente
        # del número de workers. Si solo hay 1 worker, usar el punto medio.
        if n_workers > 1:
            puct = 0.5 + i * ((1.4 - 0.5) / (n_workers - 1))
        else:
            puct = 1.4

        p = mp.Process(
            target=mcts_worker_persistent, # Nueva función
            args=(batcher_q, mcts_result_q, worker_response_queues[i], worker_task_in_queues[i], i, puct)
        )
        p.daemon = True
        p.start()

    while True:
        task = task_q.get()
        request_mcts_tree = bool(task[3]) if len(task) > 3 else False
        #print("Task listener received task:", task)
        board_demo = chess.Board(task[1])
        print(board_demo.unicode().replace("⭘", "·"))
        start = time.perf_counter()
        for q in worker_task_in_queues:
            q.put(task)

        results = [mcts_result_q.get() for _ in range(n_workers)]
        
        # Filtrar resultados None (jaque mate o errores)
        valid_results = [r for r in results if r[1] is not None]
        
        if not valid_results:
            print(f"\033[1;31m⚠️  CHECKMATE - No valid moves available\033[0m")
            tasks_result_q.put((task[0], None, 0, [], None))
            print("\033[1;33m" + "━" * 65 + "\033[0m\n")
            continue

        # 1. Acumular todas las visitas en un contador global
        total_visits = Counter()
        first_worker_initial_moves = valid_results[0][3] if valid_results else []

        for _, _, move_counts, _, *_ in valid_results:
            for move, visits in move_counts:
                total_visits[move] += visits

        best_move_final = None
        total_score_final = 0
        alternatives = []
        mcts_tree_json = valid_results[0][4] if (request_mcts_tree and len(valid_results[0]) > 4) else None

        # 2. Determinar la jugada ganadora (la que tiene más visitas totales)
        if total_visits:
            # 2. Determinar la jugada ganadora real
            top_moves = total_visits.most_common(7) 
            best_move_final = top_moves[0][0]  
            total_score_final = top_moves[0][1] 
            
            # Alternatives son las siguientes mejores jugadas
            alternatives = top_moves[1:]

            # 3. Imprimir Best Move
            print(f"\033[1;32m🏆 Best move:\033[0m {best_move_final}  \033[1;33m({total_score_final} visits)\033[0m")

            # 4. Imprimir Initial Moves (usando la referencia del primer worker)
            if first_worker_initial_moves:
                sorted_initial = sorted(first_worker_initial_moves, key=lambda x: x[1], reverse=True)
                formatted_initial = []
                for move, prob in sorted_initial:
                    color = "\033[1;32m" if prob > 0.5 else "\033[1;36m" if prob > 0.2 else "\033[0;90m"
                    formatted_initial.append(f"{move} {color}({prob:.3f})\033[0m")
                print(f"NN's Intuition: {' | '.join(formatted_initial)}")

            # 5. Imprimir Alternatives (Visitas)
            if alternatives:
                max_v = max(v for _, v in alternatives)
                formatted_alts = []
                for move, visits in alternatives:
                    ratio = visits / max_v
                    color = "\033[1;32m" if ratio > 0.8 else "\033[1;36m" if ratio > 0.4 else "\033[0;34m"
                    formatted_alts.append(f"{move} {color}[{visits}v]\033[0m")
                print(f"Other explored: {' ⮕  '.join(formatted_alts)}")

            # --- CORRECCIÓN DE LA ALERTA ---
            # Comparamos el top 1 de la red vs el top 1 del MCTS
            best_initial_move = max(first_worker_initial_moves, key=lambda x: x[1])[0] if first_worker_initial_moves else None
            
            if best_initial_move and best_initial_move != best_move_final:
                print(f"\033[1;43;30m ⚠️  MCTS CORRECTION \033[0m Red preferred \033[1;31m{best_initial_move}\033[0m but Search chose \033[1;32m{best_move_final}\033[0m")
                print("\033[1;33m" + "━" * 65 + "\033[0m")
                
        tasks_result_q.put((task[0], best_move_final, total_score_final, alternatives, mcts_tree_json)) 
        # Mostrar el arreglo de visitas por movimiento y su suma total
        total_sum = sum(total_visits.values())
        #print(f"\033[1;34m🔢 Total visits per move:\033[0m {dict(total_visits)}")
        #print(f"\033[1;34m🔢 Sum of all visits:\033[0m {total_sum}")
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
    cert_dir = Path(os.getenv("SSL_CERT_DIR", "certs"))
    cert_file = cert_dir / os.getenv("SSL_CERT_FILE", "server.crt")
    key_file = cert_dir / os.getenv("SSL_KEY_FILE", "server.key")

    http_port = int(os.getenv("HTTP_PORT", "8000"))
    https_port = int(os.getenv("HTTPS_PORT", "8443"))

    ensure_self_signed_certificate(cert_file, key_file)
    run_http_and_https(api_app, http_port, https_port, cert_file, key_file)


def ensure_self_signed_certificate(cert_file: Path, key_file: Path):
    if cert_file.exists() and key_file.exists():
        print(f"🔐 Certificado existente: {cert_file} / {key_file}")
        return

    cert_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048",
        "-keyout", str(key_file),
        "-out", str(cert_file),
        "-days", "365",
        "-subj", "/C=ES/ST=Valencia/L=Valencia/O=ChessAIThon/CN=localhost",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Certificado autofirmado generado en {cert_file}")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se encontró 'openssl'. Instálalo para generar certificados autofirmados."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Error generando certificado autofirmado: {exc.stderr.strip() or exc}"
        ) from exc


def run_http_and_https(api_app, http_port: int, https_port: int, cert_file: Path, key_file: Path):
    def run_http():
        uvicorn.run(api_app, host="0.0.0.0", port=http_port)

    http_thread = threading.Thread(target=run_http, daemon=True)
    http_thread.start()
    print(f"🌐 HTTP activo en http://0.0.0.0:{http_port}")

    print(f"🔒 HTTPS activo en https://0.0.0.0:{https_port}")
    uvicorn.run(
        api_app,
        host="0.0.0.0",
        port=https_port,
        ssl_certfile=str(cert_file),
        ssl_keyfile=str(key_file),
    )






def run():
    print("Starting main process...", __name__)
    get_model()
    main()