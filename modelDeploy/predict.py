import sys
from chessgamemultithread import chessmarro_mcts_predict_chess_move
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory

from chessmodel import init_model



if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)
    model, device = init_model()

    if len(sys.argv) < 2:
        print("Uso: python predict.py <FEN> [simulations]")
        sys.exit(1)

    fen = sys.argv[1]
    simulations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    move = chessmarro_mcts_predict_chess_move(fen, simulations,model,device)
    print(f"Mejor movimiento según MCTS ({simulations} simulaciones): {move}")
