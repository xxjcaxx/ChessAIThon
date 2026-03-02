import sys
from chessgamemultithread import chessmarro_mcts_predict_chess_move
import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory

from chessmodel import init_model



if __name__ == "__main__":

    # 2. Configuración del Entorno Multiprocesamiento
    # Establece el método de inicio de procesos a 'spawn'. 
    # Esto es OBLIGATORIO cuando se usa PyTorch con CUDA en sistemas basados en Unix 
    # para evitar problemas de "deadlocks" o corrupción de datos al bifurcar procesos.

    mp.set_start_method('spawn', force=True)

    # 3. Inicialización del Modelo y Dispositivo
    # Llama a una función que se encarga de cargar el modelo de la red neuronal 
    # y determinar el dispositivo de cómputo (CPU o, en tu caso, 'cuda'/'GPU').

    model, device = init_model()

    # 4. Manejo de Argumentos de Línea de Comandos
    
    # Verifica que se haya proporcionado al menos el argumento FEN.
    if len(sys.argv) < 2:
        print("Uso: python predict.py <FEN> [simulations]")
        sys.exit(1)

    fen = sys.argv[1]
    simulations = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # 5. Ejecución de la Predicción MCTS
    
    # Llama a la función principal que ejecuta el MCTS. 
    # Esta función recibe:
    # - fen: La posición inicial.
    # - simulations: El número de iteraciones de MCTS.
    # - model: El modelo de red neuronal cargado.
    # - device: El dispositivo ('cuda') donde se ejecuta el cómputo.

    move, json = chessmarro_mcts_predict_chess_move(fen, simulations,model,device, num_workers=64)
    print(f"Mejor movimiento según MCTS ({simulations} simulaciones): {move}")
    #print(f"Árbol MCTS en formato JSON: {json}")