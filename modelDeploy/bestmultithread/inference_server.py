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


def predict_ordered_moves_batch(boards_tensor, temperature, model):
    B = boards_tensor.size(0)

    with torch.no_grad():
        logits = model(boards_tensor)  # Output de la red [B, 4096]

    # --- MÁSCARA LEGAL ---
    # Extraemos la máscara de movimientos legales del tensor (últimos 64 canales)
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).bool()

    # --- ESTABILIZACIÓN Y MÁSCARA ---
    finfo = torch.finfo(logits.dtype)
    negbig = finfo.min / 4
    logits = torch.nan_to_num(logits, nan=0.0, posinf=finfo.max/4, neginf=finfo.min/4)
    
    # Aplicamos la máscara: los movimientos ilegales tendrán un valor bajísimo
    masked = logits.masked_fill(~legal_masks, negbig)

    # --- SOFTMAX (Probabilidades) ---
    # Usamos la temperatura para suavizar o acentuar la confianza de la red
    probs = torch.softmax(masked / temperature, dim=1)

    all_results = []

    # Iteramos sobre el batch para extraer y ordenar las jugadas legales
    for b in range(B):
        board_probs = probs[b]
        board_mask = legal_masks[b]

        # Extraemos solo los índices de los movimientos que son legales
        legal_indices = torch.nonzero(board_mask).squeeze(1)
        
        # Obtenemos las probabilidades de esos movimientos legales
        legal_probs = board_probs[legal_indices]

        # Ordenamos de mayor a menor probabilidad
        sorted_probs, sorted_order = torch.sort(legal_probs, descending=True)
        sorted_indices = legal_indices[sorted_order]

        # Convertimos a formato (UCI, Score)
        moves_with_scores = [
            (number_to_uci(int(idx)), float(score)) 
            for idx, score in zip(sorted_indices, sorted_probs)
        ]
        
        all_results.append(moves_with_scores)

    return all_results

def predict_with_value_batch(boards_tensor, temperature, model):
    B = boards_tensor.size(0)

    with torch.no_grad():
        # CAMBIO 1: El modelo es Dual-Head, ahora capturamos las dos salidas
        # policy_logits: [B, 4096], value_out: [B, 1]
        policy_logits, value_out = model(boards_tensor)

    # --- MÁSCARA LEGAL (Sin cambios) ---
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).bool()

    # --- ESTABILIZACIÓN Y MÁSCARA (Sin cambios) ---
    finfo = torch.finfo(policy_logits.dtype)
    negbig = finfo.min / 4
    logits = torch.nan_to_num(policy_logits, nan=0.0, posinf=finfo.max/4, neginf=finfo.min/4)
    masked = logits.masked_fill(~legal_masks, negbig)

    # --- SOFTMAX ---
    probs = torch.softmax(masked / temperature, dim=1)

    all_results = []

    for b in range(B):
        board_probs = probs[b]
        board_mask = legal_masks[b]

        # Extraemos índices y probabilidades legales
        legal_indices = torch.nonzero(board_mask).squeeze(1)
        legal_probs = board_probs[legal_indices]

        # Ordenamos de mayor a menor probabilidad
        sorted_probs, sorted_order = torch.sort(legal_probs, descending=True)
        sorted_indices = legal_indices[sorted_order]

        # Convertimos a formato UCI
        # Mantenemos las probabilidades (scores) porque son útiles para el MCTS
        ordered_moves = [
            (number_to_uci(int(idx)), float(score)) 
            for idx, score in zip(sorted_indices, sorted_probs)
        ]
        
        # CAMBIO 2: Extraemos el valor escalar de la evaluación para este tablero
        # El valor suele estar en rango [-1, 1] o [0, 1] según tu función de activación final
        current_value = float(value_out[b].item())

        # CAMBIO 3: Retornamos un diccionario o tupla que incluya el Value
        all_results.append({
            'moves': ordered_moves,
            'value': current_value
        })

    return all_results


def inference_server(request_q, response_q, model, device, batch_size=32, timeout=0.05):
    print("[GPU] Inference server started")
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
            preds = predict_with_value_batch(boards_tensor, 1.2, model)
            #print(preds)
            results = list(zip(task, preds)) 

            """[
    (
        # Task original (item[0] es el ID, item[1] son los datos del tablero)
        (42, ("algun_metadato", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")), 
        
        # Predicción (el diccionario 'preds' generado por la función modificada)
        {
            'moves': [
                ('Bb5', 0.4521),  # Apertura Ruy López (jugada favorita)
                ('Bc4', 0.2105),  # Apertura Italiana
                ('d4', 0.1280),   # Apertura Escocesa
                # ... resto de movimientos legales ordenados
            ],
            'value': 0.15  # La red cree que las blancas están ligeramente mejor (+0.15)
        }
    )
]""" 

            response_q.put(results)
            boards = []
            #print(results)

    print("[GPU] Shutdown")
