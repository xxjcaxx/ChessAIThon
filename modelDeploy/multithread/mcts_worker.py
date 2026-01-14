# mcts_worker.py
# mcts_worker.py
from chessgame_multiprocess import run_mcts_once

def mcts_worker(task_q, inference_q, inference_response_q, result_q):
    while True:
        task = task_q.get()
        if task is None:
            break

        task_id, fen, simulations = task

        move = run_mcts_once(
            fen,
            simulations,
            inference_q,
            inference_response_q,
        )

        result_q.put((task_id, move))
