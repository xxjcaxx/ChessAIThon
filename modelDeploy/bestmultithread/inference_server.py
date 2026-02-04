import torch
import chess
import sys
sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr



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

import torch
import gc
from queue import Empty

def inference_server(request_q, response_q, model, device, batch_size=32, timeout=0.05):
    print("[GPU] Inference server started")
    model.to(device)
    model.eval()

    while True:
        try:
            task = request_q.get(timeout=timeout)
            if task is None: break
            
            # 1. Usamos torch.no_grad() para ahorrar muchísima memoria
            with torch.no_grad():
                boards_cpu = []
                for item in task:
                    board_fen = item[1][1]
                    # Procesamos en CPU primero
                    tensor = concat_fen_legal(board_fen).to(torch.float32)
                    boards_cpu.append(tensor)
                
                if not boards_cpu:
                    continue

                # 2. Movimiento en bloque a GPU (Eficiencia de VRAM)
                boards_tensor = torch.stack(boards_cpu).to(device, non_blocking=True)
                
                # Inferencia
                preds = predict_with_value_batch(boards_tensor, 1.2, model)
                
                # 3. Conversión AGRESIVA a tipos nativos para liberar la GPU
                # Suponiendo que preds es una lista de dicts con tensores
                sanitized_preds = []
                for p in preds:
                    sanitized_p = {
                        'moves': [(m, float(s)) for m, s in p['moves']],
                        'value': float(p['value'].item()) if torch.is_tensor(p['value']) else float(p['value'])
                    }
                    sanitized_preds.append(sanitized_p)

                results = list(zip(task, sanitized_preds))
                response_q.put(results)

        except Empty:
            continue
        except Exception as e:
            print(f"[GPU ERROR] {e}")
        finally:
            # 4. Limpieza explícita de referencias pesadas en cada ciclo
            if 'boards_tensor' in locals():
                del boards_tensor
            if 'boards_cpu' in locals():
                del boards_cpu
            # Opcional: solo si notas que la RAM no baja
            # torch.cuda.empty_cache() 

    print("[GPU] Shutdown")   

"""
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

 

            response_q.put(results)
            boards = []
            #print(results)

    print("[GPU] Shutdown")
"""
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