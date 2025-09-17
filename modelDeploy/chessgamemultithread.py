import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
from chessmodel import init_model, predict_chess_move
import json
import sys

sys.path.append("./chessintionlib")  
from chess_aux_c import uci_to_number, number_to_uci, concat_fen_legal, concat_fen_legal_bits, concat_fen_legal_ptr

import multiprocessing as mp
from multiprocessing import Process, Queue, shared_memory
from queue import Empty
from multiprocessing import Manager


# mp.set_start_method('spawn', force=True)

def predict_chess_moves_vectorized(boards_tensor, temperature, model):
    """
    Predice movimientos para un batch de posiciones con Conv2D, usando
    movimientos legales codificados en los últimos 64 canales/elementos.
    """
    B = boards_tensor.size(0)
    device = boards_tensor.device

    with torch.no_grad():
        outputs = model(boards_tensor)  # [B, 4096], conv2d espera [B, C, H, W]

    # Extraer máscaras legales de los últimos 64 canales
    # Flatten del canal y el tablero a 4096
    legal_masks = boards_tensor[:, -64:, :, :].reshape(B, 4096).to(dtype=torch.bool)

    # Poner -inf en posiciones ilegales
    masked_logits = torch.where(legal_masks, outputs, -float('inf'))

    # Softmax con temperatura
    probs = torch.softmax(masked_logits / temperature, dim=1)  # [B, 4096]

    # Samplear movimiento
    move_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]

    # Convertir a UCI
    pred_moves = [number_to_uci(idx.item()) for idx in move_indices]

    return pred_moves

     

def batch_predict_worker(input_queue, output_queue, model, device):
    """
    Worker que recibe batches de boards, construye el tensor completo y predice movimientos.
    """
    model = model.to(device)
    model.eval()

    while True:
        item = input_queue.get()
        if item is None:
            break

        boards, ids = item  # boards = [tensor], ids = [task_id]
        boards_tensor = torch.stack(boards).to(device)

        preds = predict_chess_moves_vectorized(boards_tensor,1.2,model)  # lista de jugadas
        results = list(zip(ids, preds))  # [(task_id, pred), ...]
        print(results)
        output_queue.put(results)


import uuid
from multiprocessing import Process, Queue

def dispatch_loop(output_queue,pending):
    print("displach loop")
    while True:
        results = output_queue.get(block=True)
        for tid, pred in results:
            print(tid,pred)
            if tid in pending:
                pending[tid].put(pred)
                del pending[tid]

class ChessBatcher:
    """
    Clase para acumular posiciones y procesarlas en batch usando shared memory.
    Soporta múltiples clientes concurrentes y mantiene la correspondencia
    entre petición y predicción.
    """
    def __init__(self, batch_size, model, device, manager):
        self.batch_size = batch_size
        self.device = device
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.worker = Process(target=batch_predict_worker,
                              args=(self.input_queue, self.output_queue, model, device))
        self.worker.start()
        self.current_batch = []
        
       
        self.manager = manager
        self.pending = self.manager.dict()  # task_id -> Queue de respuesta
        
        self.dispatch_loop_process = Process(target=dispatch_loop, args=(self.output_queue,self.pending), daemon=True)
        self.dispatch_loop_process.start()


    def add_board(self, board_tensor, response_q=None):
        if response_q is None:
            response_q = self.manager.Queue()
        task_id = uuid.uuid4().hex
        self.pending[task_id] = response_q
        self.current_batch.append((board_tensor, task_id))
        if len(self.current_batch) >= self.batch_size:
            self._flush_batch()
        return response_q

    def _flush_batch(self):
        """
        Envía el batch acumulado al worker y limpia la lista.
        """
        if self.current_batch:
            #print(self.current_batch)
            boards, ids = zip(*self.current_batch)
            self.input_queue.put((boards, ids))
            self.current_batch = []


    def poll_predictions(self):
        """
        Procesa todo lo que haya llegado en output_queue
        y despacha a las colas correspondientes.
        """

        while True:
            results = self.output_queue.get(block=True)
            for tid, pred in results:
                if tid in self.pending:
                    self.pending[tid].put(pred)
                    del self.pending[tid]


        print("poll_predictions finalizado")

    def close(self):
        """
        Cierra el worker correctamente.
        """
        self._flush_batch()
        self.input_queue.put(None)  # Señal de cierre
        self.worker.join()



def chessmarro_mcts_predict_chess_move(fen, simulations, model, device):

    best_move = 1234




####################################3

    fens = [
        "r3k1nr/ppp2ppp/2np1q2/2b5/2Q1PB2/7P/PPP2P1P/RN2KB1R b KQkq - 0 8",
        "r1bqkbnr/pppppppp/n7/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 2",
        '5k2/R7/3K4/4p3/5P2/8/8/5r2 w - - 0 0',
    '5k2/1R6/4p1p1/1pr3Pp/7P/1K6/8/8 w - - 0 0',
    '5k2/8/p7/4K1P1/P4R2/6r1/8/8 b - - 0 0',
    '8/8/8/p2r1k2/7p/PP1RK3/6P1/8 b - - 0 0',
    '8/8/8/1P4p1/5k2/5p2/P6K/8 b - - 0 0',
    '3b2k1/1p3p2/p1p5/2P4p/1P2P1p1/5p2/5P2/4RK2 w - - 0 0',
    '5k2/3R4/2K1p1p1/4P1P1/5P2/8/3r4/8 b - - 0 0',
    '6k1/6pp/5p2/8/5P2/P7/2K4P/8 b - - 0 0',
    '8/3R4/8/r3N2p/P1Pp1P2/2k2K1P/3r4/8 w - - 0 0',
    '6k1/8/6r1/8/5b2/2PR4/4K3/8 w - - 0 0',
    '8/1p3k2/3B4/8/3b2P1/1P6/6K1/8 b - - 0 0',
    '8/8/8/2p1k3/P6R/1K6/6rP/8 w - - 0 0',
    '6k1/5p1p/6p1/1P1n4/1K4P1/N6P/8/8 w - - 0 0',
    '8/k5r1/2N5/PK6/2B5/8/8/8 b - - 0 0',
    '6k1/8/5K2/8/5P1R/r6P/8/8 b - - 0 0',
    '8/8/4k1KP/p5P1/r7/8/8/8 w - - 0 0',
    '1R6/p2r4/2ppkp2/6p1/2PKP2p/P4P2/6PP/8 b - - 0 0',
    '8/7p/6p1/8/k7/8/2K3P1/8 b - - 0 0',
    'R7/8/8/6p1/4k3/3rPp1P/8/6K1 b - - 0 0',
    '8/7p/1p1k2p1/p1p2p2/8/PP2P2P/4KPP1/8 w - - 0 0' 
    ]

    # Preprocesar todas las posiciones
    boards = [concat_fen_legal(fen).cpu() for fen in fens]  # cada uno es numpy
    manager = Manager()
    batcher = ChessBatcher(8, model, 'cuda:0', manager=manager)



    res_queues = []

    for board_tensor in boards:
        # Creamos una cola compartida por manager para cada petición
        response_q = manager.Queue()
        
        # Añadimos el board al batcher pasando la cola de respuesta
        batcher.add_board(board_tensor, response_q=response_q)
        
        # Guardamos la cola para que el cliente pueda esperar su predicción
        res_queues.append(response_q)

    # Enviar el batch final que quede
    batcher._flush_batch()

    # Recuperar resultados de cada cola
    for i, q in enumerate(res_queues):
        print(f"Esperando predicción en cola {i}:", q)
        move = q.get()  # se desbloquea automáticamente cuando el worker termine
        print(f"Recibido {i}:", move)


    # Cerrar worker
    batcher.close()


#######################################

    return best_move