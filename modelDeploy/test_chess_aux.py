import time
import chess
import numpy as np

# Lista de FENs de ejemplo (puedes añadir muchas más)
fens = [
    "r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8",
    "r1bqk2r/pppp1ppp/2n2n2/2b1P3/2Bp4/2P2N2/PP3PPP/RNBQK2R b KQkq - 0 6",
    "r1bq1rk1/1p2bppp/p1pp1n2/8/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
] * 50  # Replicar para 200 FENs

# --- Version original (importa tu concat_fen_legal original) ---
from chess_aux import concat_fen_legal as concat_fen_old  # Ajusta import según tu fichero

# --- Version optimizada ---
from chess_aux_optim import concat_fen_legal as concat_fen_new  # Ajusta import según tu fichero

# --- Benchmark función ---
def benchmark(fens, func, name):
    start = time.time()
    results = [func(fen) for fen in fens]
    end = time.time()
    total_time = end - start
    print(f"{name}: total={total_time:.4f}s, per FEN={total_time/len(fens):.6f}s")
    return results

# Ejecutar benchmark
print("Running benchmark...")

res_old = benchmark(fens, concat_fen_old, "Original concat_fen_legal")
res_new = benchmark(fens, concat_fen_new, "Optimized concat_fen_legal")


import numpy as np

def deep_compare_fens(fens, func1, func2):
    mismatches = 0
    for i, fen in enumerate(fens):
        res1 = func1(fen)
        res2 = func2(fen)
        if not np.array_equal(res1, res2):
            print(f"\033[1;31mMismatch at FEN {i}:\033[0m {fen}")
            mismatches += 1
    if mismatches == 0:
        print("\033[1;32mAll FENs match perfectly!\033[0m")
    else:
        print(f"\033[1;33mTotal mismatches: {mismatches}/{len(fens)}\033[0m")


deep_compare_fens(fens, concat_fen_old, concat_fen_new)