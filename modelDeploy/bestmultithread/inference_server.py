import torch
import chess
import sys
sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr

def predict_chess_moves_vectorized(boards_tensor, temperature, model):
    # Get the batch size, which is the number of boards to process.
    B = boards_tensor.size(0)

    # Disable gradient calculations to save memory and computation time during inference.
    with torch.no_grad():
        # Pass the batch of board features through the neural network to get raw output logits.
        logits = model(boards_tensor)  # [B,4096]

    # --- LEGAL MASK ---
    # Extract the 64-channel legal move mask from the board tensor (assuming it's the last 64 channels).
    # Flatten the 64x8x8 mask into a 4096-element boolean vector for each board.
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).bool()

    # --- SANITIZE LOGITS  ---
    # Get the floating-point information for the logits' data type (e.g., float32).
    finfo = torch.finfo(logits.dtype)
    # Define a numerically stable, large negative value for masking illegal moves.
    negbig = finfo.min / 4
    # Replace any NaN, posinf, or neginf values in the logits to ensure numerical stability.
    logits = torch.nan_to_num(logits, nan=0.0,
                              # Clamp positive infinities to a large but stable number.
                              posinf=finfo.max/4,
                              # Clamp negative infinities to a small but stable number.
                              neginf=finfo.min/4)

    # --- MASK: Set illegal moves to a very low value ---
    # Set the logits for all illegal moves to the large negative value (`negbig`).
    masked = logits.masked_fill(~legal_masks, negbig)

    # --- SOFTMAX ---
    # Apply temperature scaling (controls exploration) and the softmax function to get move probabilities.
    probs = torch.softmax(masked / temperature, dim=1)

    # --- HANDLE INVALID ROWS (e.g., Checkmate/Stalemate) ---
    # Ensure all resulting probabilities are non-negative due to potential floating-point underflow.
    probs = torch.clamp(probs, min=0.0)
    # Calculate the sum of probabilities for each board in the batch.
    row_sums = probs.sum(dim=1, keepdim=True)
    # Identify boards where the probability sum is valid (greater than zero and finite).
    valid = (row_sums > 0) & torch.isfinite(row_sums)

    # If the probability sum is zero (invalid row), prepare a uniform distribution over all legal moves.
    # Create a float tensor from the legal mask for the uniform distribution fallback.
    uniform = legal_masks.float()
    # Normalize the uniform mask so that the probability sum of legal moves is exactly 1.
    uniform = uniform / uniform.sum(dim=1, keepdim=True).clamp(min=1)

    # For valid rows, use the normalized predicted probabilities; otherwise, use the uniform distribution fallback.
    final_probs = torch.where(valid, probs / row_sums, uniform)

        # --- SAMPLE (Standard MCTS Mode) ---
        # Draw one sample move index from the distribution for each board.
        # torch.multinomial requires a 2D tensor.
    idxs = torch.multinomial(final_probs, num_samples=1).squeeze(1)

        # --- Convert to UCI ---
        # Convert the selected move indices into Universal Chess Interface (UCI) string format.
        # The function 'number_to_uci' is assumed to be accessible.
    return [number_to_uci(int(i)) for i in idxs]


def inference_server(request_q, response_q, model, device, batch_size=32, timeout=0.05):
    
    model.eval()

    boards = []

    while True:
        #print("[GPU] Waiting for tasks...")
        try:
            task = request_q.get(timeout=timeout)
        except Exception:
            task = None
        if task is not None:
            #print("[GPU] Received task:", task)
            #response_q.put((task[0], "e2e4"))  # Dummy response
            #continue
            for item in task:
                board_fen = item[1][1]
                #board = chess.Board(board_fen)
                board_tensor = concat_fen_legal(board_fen).to(device)
                board_tensor = board_tensor.to(torch.float32)
                boards.append(board_tensor)  # Store the id and board
       
            boards_tensor = torch.stack(boards)
            boards_tensor = boards_tensor.to(device)
            preds = predict_chess_moves_vectorized(boards_tensor, 1.2, model)
            results = list(zip(task, preds))  # [((id_worker, (id_thread, fen) ), move)]  [((23, (3, 'rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 3')), 'f3e5'), ((23, (1, 
            response_q.put(results)
            boards = []
            #print(results)

    print("[GPU] Shutdown")
