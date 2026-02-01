# chessgame_multiprocess.py
# ============================================================
# Versión LIMPIA y multiproceso correcta del motor de ajedrez
# ============================================================

import uuid
import time
import chess
import random
import math

from chess_aux_c import concat_fen_legal


# ============================================================
# IPC CLIENTE DE INFERENCIA (NO MODELO AQUÍ)
# ============================================================

class InferenceClient:
    """
    Cliente ligero para pedir inferencias al proceso GPU.
    NO crea colas.
    NO conoce el modelo.
    """

    def __init__(self, request_q, response_q, timeout=10.0):
        self.request_q = request_q
        self.response_q = response_q
        self.timeout = timeout

    def predict_move(self, fen: str) -> str:
        task_id = uuid.uuid4().hex
        board_tensor = concat_fen_legal(fen)

        self.request_q.put((task_id, board_tensor))

        start = time.time()
        while True:
            rid, move = self.response_q.get()
            if rid == task_id:
                return move

            if time.time() - start > self.timeout:
                raise TimeoutError("Inference timeout")


# ============================================================
# MCTS NODE
# ============================================================

class MCTSNode:
    __slots__ = (
        "state",
        "parent",
        "move",
        "children",
        "visits",
        "value",
    )

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) == self.state.legal_moves.count()

    def best_child(self, c=1.4):
        best_score = -1e9
        best = None

        for child in self.children:
            if child.visits == 0:
                return child

            uct = (
                child.value / child.visits
                + c * math.sqrt(math.log(self.visits) / child.visits)
            )

            if uct > best_score:
                best_score = uct
                best = child

        return best


# ============================================================
# MCTS CORE (CPU PURO)
# ============================================================

class MCTS:
    def __init__(self, root, inference_client, simulations=100):
        self.root = root
        self.client = inference_client
        self.simulations = simulations

    def search(self):
        for _ in range(self.simulations):
            node = self._select()
            result = self._simulate(node)
            self._backpropagate(node, result)

        best = max(self.root.children, key=lambda c: c.visits, default=None)
        return best.move if best else None

    def _select(self):
        node = self.root
        while not node.state.is_game_over():
            if not node.is_fully_expanded():
                return self._expand(node)
            node = node.best_child()
        return node

    def _expand(self, node):
        tried = {child.move for child in node.children}

        for move in node.state.legal_moves:
            if move not in tried:
                new_state = node.state.copy()
                new_state.push(move)
                child = MCTSNode(new_state, node, move)
                node.children.append(child)
                return child

        return node

    def _simulate(self, node):
        state = node.state.copy()
        depth = 0

        while not state.is_game_over() and depth < 60:
            try:
                move_uci = self.client.predict_move(state.fen())
                move = chess.Move.from_uci(move_uci)
                if move not in state.legal_moves:
                    raise ValueError
            except Exception:
                move = random.choice(list(state.legal_moves))

            state.push(move)
            depth += 1

        result = state.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0

    def _backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent


# ============================================================
# FUNCIÓN DE ALTO NIVEL (IMPORTABLE)
# ============================================================

def run_mcts_once(
    fen: str,
    simulations: int,
    inference_request_q,
    inference_response_q,
):
    """
    Función segura para ejecutar UNA predicción MCTS.
    Ideal para llamar desde procesos worker.
    """

    board = chess.Board(fen)
    root = MCTSNode(board)

    client = InferenceClient(
        inference_request_q,
        inference_response_q,
    )

    mcts = MCTS(
        root=root,
        inference_client=client,
        simulations=simulations,
    )

    best_move = mcts.search()
    return str(best_move) if best_move else None
