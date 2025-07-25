def predict_chess_move(fen_position):
    """
    Given a FEN chess board position, return the best move prediction.
    """
    board = concat_fen_legal(fen_position)
    board_matrix = torch.tensor(board, dtype=torch.float32)
    board_matrix = board_matrix.unsqueeze(0)
    board_matrix = board_matrix.to(device)
    with torch.no_grad():
        outputs = model(board_matrix)
    # We need always mask because not all probabilities will be legal. 
    legal_moves_mask = board[-64:]
    legal_moves_mask = torch.tensor(legal_moves_mask.reshape(4096), dtype=torch.float32).to(device)
    outputs = outputs * legal_moves_mask
    # This is necessary for softmax to not choose a non legal move
    outputs = outputs.masked_fill(legal_moves_mask == 0, -float('inf'))

    # Apply softmax with temperature for more diverse move selection
    temperature = 1.2  # Adjust temperature as needed
    probabilities = torch.softmax(outputs / temperature, dim=1)  # Softmax to avoid repetitions
    move_index = torch.multinomial(probabilities, 1).item()
    return number_to_uci(move_index)

def get_dynamic_top_moves(fen_position, min_moves=3, max_moves=10):
    """
    Devuelve una cantidad dinámica de los mejores movimientos según la evaluación de la IA.
    Si la posición es clara, devuelve menos movimientos. Si es incierta, devuelve más.
    """
    board = concat_fen_legal(fen_position)
    board_matrix = torch.tensor(board, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(board_matrix)

    # Aplicar máscara de movimientos legales
    legal_moves_mask = torch.tensor(board[-64:].reshape(4096), dtype=torch.float32).to(device)
    outputs = outputs.masked_fill(legal_moves_mask == 0, -float('inf'))

    # Aplicar softmax para obtener probabilidades
    probabilities = torch.softmax(outputs, dim=1)

    # Filtrar solo movimientos legales
    legal_indices = torch.nonzero(legal_moves_mask, as_tuple=True)[0]
    legal_probs = probabilities[0][legal_indices]  # Extrae solo los valores legales

    # Obtener los mejores movimientos legales
    num_legal_moves = len(legal_probs)
    top_k = min(num_legal_moves, max_moves)  # Asegurar que no se pidan más de los disponibles

    if top_k == 0:
        return []  # No hay movimientos legales, retornar lista vacía

    top_values, top_indices = torch.topk(legal_probs, top_k, largest=True)

    # Normalizar las probabilidades
    top_probs = top_values.cpu().numpy()
    prob_ratio = top_probs[0] / (top_probs[1] + 1e-6) if len(top_probs) > 1 else 1.0  # Evitar división por 0

    # Si el mejor movimiento es muy superior, reducimos la exploración
    num_moves = min_moves if prob_ratio > 2.0 and top_probs[0] > 0.6 else top_k

    # Convertir los índices a UCI
    top_moves_uci = [number_to_uci(legal_indices[idx].item()) for idx in top_indices[:num_moves]]
    print(top_moves_uci, fen_position)

    return top_moves_uci
