import ctypes
import numpy as np
import cProfile
import pstats

chess_extension = ctypes.CDLL('./c/chessintionlib/libchessintionlib.so') 


chess_extension.concat_fen_legal.argtypes = [ctypes.c_char_p]
#chess_extension.concat_fen_legal.restype = ctypes.POINTER(ctypes.c_int * 77*8*8)  # This should return a pointer to an array
chess_extension.concat_fen_legal.restype = ctypes.POINTER(ctypes.c_uint8 * 616)

chess_extension.concat_fen_legal_bits.argtypes = [ctypes.c_char_p]
chess_extension.concat_fen_legal_bits.restype = ctypes.POINTER(ctypes.c_uint8 * 616)

def concat_fen_legal(fen):
    fen_bytes = fen.encode('utf-8')
    # Call concat_fen_legal from the shared library
    result_ptr = chess_extension.concat_fen_legal(fen_bytes)
    compressed_array = np.frombuffer(result_ptr.contents, dtype=np.uint8)
    bit_array = np.unpackbits(compressed_array).astype(np.uint8)
    array_np = bit_array.reshape(77, 8, 8)
    return array_np
    #np.set_printoptions(threshold=np.inf)
    #print(array_np)

#concat_fen_legal('3k4/1K3B2/1BP5/1n3p2/7p/7P/7b/8 b - - 4 3')

def concat_fen_legal_bits(fen):
    fen_bytes = fen.encode('utf-8')
    
    result_ptr = chess_extension.concat_fen_legal_bits(fen_bytes)
    compressed_array = np.frombuffer(result_ptr.contents, dtype=np.uint8)
    bit_array = np.unpackbits(compressed_array).astype(np.uint8)
    array_np = bit_array.reshape(77, 8, 8)
    #np.set_printoptions(threshold=np.inf)
    #print(array_np)

    return array_np



#concat_fen_legal_bits('3k4/1K3B2/1BP5/1n3p2/7p/7P/7b/8 b - - 4 3')
fen_collection = ['5k2/R7/3K4/4p3/5P2/8/8/5r2 w - - 0 0',
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
'8/7p/1p1k2p1/p1p2p2/8/PP2P2P/4KPP1/8 w - - 0 0']

def bench(n):
    for i in range(n):
        for j in fen_collection: 
            a = concat_fen_legal_bits(j)
            b = concat_fen_legal(j)
            print(np.array_equal(a, b))

cProfile.run('bench(2)', 'output.prof')

# Print sorted profiling results (100 simulations) 19.466 seconds
p = pstats.Stats('output.prof')
p.strip_dirs().sort_stats('cumulative').print_stats(20)  # Top 10 slowest functions

fen = '5k2/R7/3K4/4p3/5P2/8/8/5r2 w - - 0 0'
a = concat_fen_legal_bits(fen)
b = concat_fen_legal(fen)

print(a[0])
print(b[0])

