#import torch
#import chess

def init_model():
    return None, 'torch.device("cpu")'

def predict_chess_moves_vectorized(boards, temperature, model):
    # Dummy implementation for illustration purposes
    batch_size = 32 #boards.size(0)
    moves = []
    for i in range(batch_size):
        moves.append("e2e4")  # Placeholder move
    return moves

def inference_server(request_q, response_q, batch_size=32, timeout=0.05):
    model, device = init_model()
    #model.eval()

    batch = []
    ids = []

    while True:
        #print("[GPU] Waiting for tasks...")
        try:
            task_id, board_tensor = request_q.get(timeout=timeout)
            batch.append(board_tensor)
            ids.append(task_id)
        except Exception:
            pass

        if batch:
            boards = batch # torch.stack(batch).to(device)
            
            moves = predict_chess_moves_vectorized(
                    boards, temperature=1.2, model=model
                )

            for tid, move in zip(ids, moves):
                response_q.put((tid, move))

            batch.clear()
            ids.clear()


    print("[GPU] Shutdown")
