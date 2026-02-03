import ctypes
import numpy as np
import torch
import os

# Load the shared library (adjust path to the .so file)
current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(current_dir, 'libchessintionlib.so')

chess_extension = ctypes.CDLL(lib_path)

# Define the function signatures of the C++ functions in the shared library

# Define matrix types
Matrix3D77 = ctypes.POINTER(ctypes.c_int)

# Define function signatures


chess_extension.board_to_3D_matrix.argtypes = [ctypes.c_void_p]
chess_extension.board_to_3D_matrix.restype = ctypes.POINTER(ctypes.c_int)

chess_extension.legal_moves_to_64_8_8.argtypes = [ctypes.c_void_p]
chess_extension.legal_moves_to_64_8_8.restype = ctypes.POINTER(ctypes.c_int)

chess_extension.uci_to_number.argtypes = [ctypes.c_char_p]
chess_extension.uci_to_number.restype = ctypes.c_int

chess_extension.number_to_uci.argtypes = [ctypes.c_int]
chess_extension.number_to_uci.restype = ctypes.c_char_p


chess_extension.concat_fen_legal.argtypes = [ctypes.c_char_p]
chess_extension.concat_fen_legal.restype = ctypes.POINTER(ctypes.c_uint8 * 616)

chess_extension.concat_fen_legal_bits.argtypes = [ctypes.c_char_p]
chess_extension.concat_fen_legal_bits.restype = ctypes.POINTER(ctypes.c_uint8 * 616)


def uci_to_number(uci_move):
    return chess_extension.uci_to_number(uci_move.encode('utf-8'))

def number_to_uci(number_move):
    return chess_extension.number_to_uci(number_move).decode('utf-8')


def concat_fen_legal(fen):
    fen_bytes = fen.encode('utf-8')
    # Call concat_fen_legal from the shared library
    result_ptr = chess_extension.concat_fen_legal(fen_bytes)

    compressed_tensor = torch.tensor(list(result_ptr.contents), dtype=torch.uint8, device="cuda")
    bit_tensor = ((compressed_tensor[:, None] >> torch.arange(8, device="cuda")) & 1).to(torch.float32)
    bit_tensor = bit_tensor.view(77, 8, 8)
    return bit_tensor

#print(uci_to_number('c2c3'))
#print(number_to_uci(50))
#print(concat_fen_legal('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))


def concat_fen_legal_bits(fen):
    fen_bytes = fen.encode('utf-8')
    board_ptr =   chess_extension.concat_fen_legal_bits(fen_bytes)
    compressed_tensor = torch.tensor(list(board_ptr.contents), dtype=torch.uint8, device="cuda")
    bit_tensor = ((compressed_tensor[:, None] >> torch.arange(8, device="cuda")) & 1).to(torch.float32)

    # Darle forma (77, 8, 8)
    return bit_tensor.view(77, 8, 8)


def concat_fen_legal_ptr(fen):
    fen_bytes = fen.encode('utf-8')
    
    # Llamar a la función de la librería compartida
    return chess_extension.concat_fen_legal_bits(fen_bytes)
    