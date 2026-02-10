import torch
import chess
import sys
sys.path.append("./chessintionlib")  
from chess_aux import uci_to_number, number_to_uci, concat_fen_legal #, #concat_fen_legal_bits, concat_fen_legal_ptr

import os
import psutil
import numpy as np



def predict_with_value_batch_fast(boards_tensor, temperature, model):
    with torch.no_grad():
        policy_logits, value_out = model(boards_tensor)

    B = boards_tensor.size(0)
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).bool()

    finfo = torch.finfo(policy_logits.dtype)
    negbig = finfo.min / 4
    logits = torch.nan_to_num(policy_logits, nan=0.0,
                              posinf=finfo.max/4, neginf=finfo.min/4)

    masked = logits.masked_fill(~legal_masks, negbig)
    probs = torch.softmax(masked / temperature, dim=1)

    # 🚀 Devuelve TODO como tensores
    return probs, legal_masks, value_out.squeeze(1)


def extract_moves(probs, legal_masks, values, topk=32):
    probs = probs.cpu()
    legal_masks = legal_masks.cpu()
    values = values.cpu()

    results = []
    for b in range(probs.size(0)):
        p = probs[b][legal_masks[b]]
        idx = torch.nonzero(legal_masks[b]).squeeze(1)

        topk_probs, topk_idx = torch.topk(p, min(topk, p.numel()))
        topk_moves = idx[topk_idx]

        results.append({
            "moves": [(number_to_uci(int(i)), float(s))
                      for i, s in zip(topk_moves.cpu(), topk_probs.cpu())],
            "value": float(values[b])
        })
    return results


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
                
                raw_boards = [item[1][1] for item in task]
                if raw_boards:
                    batch_np = np.array(raw_boards)
                    boards_tensor = torch.from_numpy(batch_np).to(device, dtype=torch.float32, non_blocking=True)
                    
                """
                # Inferencia
                preds = predict_with_value_batch(boards_tensor, 1.2, model)
                

                results = list(zip(task, preds))
               # print(f"Resultados preparados para enviar al worker: {results}")
                response_q.put(results)
                """
                                # 🔥 GPU inference (solo tensores)
                probs, legal_masks, values = predict_with_value_batch_fast(
                    boards_tensor, 1.2, model
                )

                # 🧠 CPU post-proceso (fuera del hot path GPU)
                preds = extract_moves(
                    probs, legal_masks, values, topk=32
                )

                # Empaquetado final
                results = list(zip(task, preds))
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
