# inference_server.py
import torch
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessmodel import init_model
sys.path.append("../chessintionlib")  
from chess_aux_c import number_to_uci
from chessgamemultithread import predict_chess_moves_vectorized

# inference_server.py
def inference_server(request_q, response_q, batch_size=32, timeout=0.05):
    model, device = init_model()
    model.eval()

    batch = []
    ids = []

    while True:
        try:
            task_id, board_tensor = request_q.get(timeout=timeout)
            batch.append(board_tensor)
            ids.append(task_id)
        except Exception:
            pass

        if batch:
            boards = torch.stack(batch).to(device)
            with torch.no_grad():
                moves = predict_chess_moves_vectorized(
                    boards, temperature=1.2, model=model
                )

            for tid, move in zip(ids, moves):
                response_q.put((tid, move))

            batch.clear()
            ids.clear()


    print("[GPU] Shutdown")
