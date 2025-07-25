#include "../chess-library-master/include/chess.hpp"
#include <map>    // Include this for std::map
#include <vector> // Needed for std::vector

extern "C"
{

  using namespace chess;

  using Matrix = std::array<std::array<int, 8>, 8>;

/*
  Matrix board_to_matrix(Board &board, PieceType piece_type)
  {
    Matrix matrix = {}; // Initialize an 8x8 matrix with zeros
                        // std::cout <<  board.getFen();
    std::uint64_t bbw = board.pieces(piece_type, Color::WHITE).getBits();
    std::uint64_t bbb = board.pieces(piece_type, Color::BLACK).getBits();

    for (int i = 0; i < 64; ++i)
    {
      // Get the row and column based on the bit index
      int row = i / 8; // Row index (0 to 7)
      int col = i % 8; // Column index (0 to 7)

      // Extract the bit at the current position
      if (((bbw >> (63 - i)) & 1) == 1)
      {
        matrix[row][col] = 1;
      }
      if (((bbb >> (63 - i)) & 1) == 1)
      {
        matrix[row][col] = -1;
      }
    }

    return matrix;
  }*/

  using Matrix3D = std::vector<Matrix>;

  Matrix3D board_to_3D_matrix(Board &board)
  {
    Matrix3D matrix3D = {};
    std::vector<PieceType> pieces = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

    for (auto piece : pieces)
    {
      std::uint64_t bbw = board.pieces(piece, Color::WHITE).getBits();
      std::uint64_t bbb = board.pieces(piece, Color::BLACK).getBits();
      Matrix white_matrix = {}, black_matrix = {};

      for (int i = 0; i < 64; ++i)
      {
        // Get the row and column based on the bit index
        int row = i / 8; // Row index (0 to 7)
        int col = i % 8; // Column index (0 to 7)

        // Extract the bit at the current position
        if (((bbw >> (63 - i)) & 1) == 1)
        {
          white_matrix[row][7-col] = 1;
        }
        if (((bbb >> (63 - i)) & 1) == 1)
        {
          black_matrix[row][7-col] = 1;
        }
      }

      matrix3D.push_back(white_matrix);
      matrix3D.push_back(black_matrix);
    }

    Matrix color_matrix = {};
    if (board.sideToMove() == Color::WHITE)
    {
      for (auto &row : color_matrix)
      {
        std::fill(row.begin(), row.end(), 1);
      }
    }
    else
    {
      for (auto &row : color_matrix)
      {
        std::fill(row.begin(), row.end(), 0);
      }
    }
    matrix3D.push_back(color_matrix);

    return matrix3D;
  }



  ////////////// Con bits

using Matrix3D_bits = std::vector<uint64_t>; 

Matrix3D_bits board_to_3D_bits_matrix(Board &board)
  {
    Matrix3D_bits matrix3D = {};
    std::vector<PieceType> pieces = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

    for (auto piece : pieces)
    {
      std::uint64_t bbw = board.pieces(piece, Color::WHITE).getBits();
      std::uint64_t bbb = board.pieces(piece, Color::BLACK).getBits();
      matrix3D.push_back(bbw);
      matrix3D.push_back(bbb);
    }

    uint64_t color_matrix = 0;
    if (board.sideToMove() == Color::WHITE)
    {
      color_matrix = ~uint64_t(0);
    }
    else
    {
      color_matrix = 0;
    }
    matrix3D.push_back(color_matrix);

    return matrix3D;
  }










  std::map<std::pair<int, int>, int> codes; // Declare the map

  void initialize_codes()
  {
    int i = 0;
    // All 56 regular moves
    for (int nSquares = 1; nSquares < 8; ++nSquares)
    {
      for (const auto &direction : std::vector<std::pair<int, int>>{
               {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}})
      {
        codes[{nSquares * direction.first, nSquares * direction.second}] = i++;
      }
    }

    // 8 Knight moves
    codes[{1, 2}] = i++;
    codes[{2, 1}] = i++;
    codes[{2, -1}] = i++;
    codes[{1, -2}] = i++;
    codes[{-1, -2}] = i++;
    codes[{-2, -1}] = i++;
    codes[{-2, 1}] = i++;
    codes[{-1, 2}] = i++;
  }

  using Matrix3DMoves = std::array<std::array<std::array<int, 8>, 8>, 64>;


  Matrix3DMoves legal_moves_to_64_8_8(const Board &board)
  {
    Matrix3DMoves array6488 = {}; // Initialize to all zeros
    initialize_codes();
    // Get legal moves
    Movelist legal_moves;
    movegen::legalmoves(legal_moves, board);

    for (auto move : legal_moves)
    { // square.rank();
    if (move.typeOf() == Move::CASTLING) {
        std::string move_str = uci::moveToUci(move);
        std::string from_str = move_str.substr(0, 2);
    std::string to_str = move_str.substr(2, 2);
        Square from_sq = Square(from_str);
    Square to_sq = Square(to_str);
        move = Move::make<Move::NORMAL>(from_sq, to_sq);
    }
   
      int from_rank = move.from().rank();
      int from_file = move.from().file();
      int to_rank = move.to().rank();
      int to_file = move.to().file();

      // Compute encoded move
      std::pair<int, int> move_vector = {to_file - from_file, to_rank - from_rank};

      // Check if move exists in `codes`
      if (codes.find(move_vector) != codes.end())
      {
        // std::cout << move << std::endl;
        int code_index = codes[move_vector];
        array6488[code_index][7 - from_rank][from_file] = 1;
      }
    }
    return array6488;
  }


  //////////// Con bits


using BitboardArray = std::array<uint64_t, 64>;

BitboardArray legal_moves_to_64_bitboards(const Board &board)
{
    BitboardArray bitboards = {}; // 64 enteros de 64 bits inicializados en 0
    initialize_codes();
    
    // Obtener movimientos legales
    Movelist legal_moves;
    movegen::legalmoves(legal_moves, board);

    for (auto move : legal_moves)
    {
        if (move.typeOf() == Move::CASTLING) {
            std::string move_str = uci::moveToUci(move);
            std::string from_str = move_str.substr(0, 2);
            std::string to_str = move_str.substr(2, 2);
            Square from_sq = Square(from_str);
            Square to_sq = Square(to_str);
            move = Move::make<Move::NORMAL>(from_sq, to_sq);
        }

        int from_rank = move.from().rank();
        int from_file = move.from().file();
        int to_rank = move.to().rank();
        int to_file = move.to().file();

        // Computar vector del movimiento
        std::pair<int, int> move_vector = {to_file - from_file, to_rank - from_rank};

        // Verificar si el movimiento está en `codes`
        if (codes.find(move_vector) != codes.end())
        {
            int code_index = codes[move_vector];
            int bit_position = (7 - from_rank) * 8 + from_file; // Mapeo 2D a 1D
            bitboards[code_index] |= (1ULL << bit_position); // Activar bit en la posición correcta
        }
    }
    return bitboards;
}








  // Movement encoding map (should be initialized globally)
  extern std::map<std::pair<int, int>, int> codes;

  int uci_to_number(const char *uci_move)
  {
    // Move m = uci::uciToMove(board, uci_move);
    // std::cout << uci_move;
    initialize_codes();

    int from_file = uci_move[0] - 'a'; // 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
    int from_rank = uci_move[1] - '1'; // '1' -> 0, '2' -> 1, ..., '8' -> 7
    int to_file = uci_move[2] - 'a';
    int to_rank = uci_move[3] - '1';

    // std::cout << uci_move << " " << to_file - from_file << " " << to_rank - from_rank;

    // Compute move vector (delta file, delta rank)
    std::pair<int, int> move_vector = {to_file - from_file, to_rank - from_rank};

    // Lookup move code
    if (codes.find(move_vector) == codes.end())
    {
      //std::cerr << "Error: Move not found in codes!" << std::endl;
      return -1;
    }
    int move_code = codes[move_vector];

    // Compute flattened index (64 * 8 * rank + 8 * file + move_code)
    int pos = (move_code * 8 * 8) + ((7 - from_rank) * 8) + from_file;

    return pos;
  }

  const char *number_to_uci(int number_move)
  {
    // Extract move_code, from_rank, and from_file from the number
    int move_code = number_move / (8 * 8);
    int from_rank = 7 - (number_move / 8) % 8; // Reverse row index
    int from_file = number_move % 8;           // File index

    // Find the corresponding move vector (delta file, delta rank)
    std::pair<int, int> code;
    for (const auto &[key, value] : codes)
    {
      if (value == move_code)
      {
        code = key;
        break;
      }
    }

    // Convert rank & file to UCI notation (e.g., e2e4)
    static char uci_move[5]; // 4 characters + null terminator
    uci_move[0] = 'a' + from_file;
    uci_move[1] = '1' + from_rank;
    uci_move[2] = 'a' + from_file + code.first;
    uci_move[3] = '1' + from_rank + code.second;
    uci_move[4] = '\0'; // Null-terminate

    return uci_move;
  }

  using Matrix3D77 = std::array<std::array<std::array<int, 8>, 8>, 77>;
  using PackedArray = std::array<uint8_t, 616>;
  PackedArray packed_data = {};

  const PackedArray* concat_fen_legal(const char *fen)
  {
    packed_data = {};
    std::string fen_string(fen);
   // std::cerr << "fen:" << fen_string << std::endl;

    chess::Board board(fen_string); // Load FEN into board

    // Get board representation (13,8,8) and legal moves (64,8,8)
    auto fen_matrix = board_to_3D_matrix(board);
    auto legal_moves = legal_moves_to_64_8_8(board);

    // Concatenate into (77,8,8)
    // Matrix3D77 fen_matrix_legal_moves = {};
    auto *fen_matrix_legal_moves = new Matrix3D77();
    // Copy 13 board layers
    for (int i = 0; i < 13; ++i)
    {
      (*fen_matrix_legal_moves)[i] = fen_matrix[i];
    }

    // Copy 64 legal move layers
    for (int i = 0; i < 64; ++i)
    {
      (*fen_matrix_legal_moves)[13 + i] = legal_moves[i];
    }


    int bit_index = 0;
    for (int i = 0; i < 77; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                int byte_pos = bit_index / 8;
                int bit_pos = bit_index % 8;

                if ((*fen_matrix_legal_moves)[i][j][k]) {
                    packed_data[byte_pos] |= (1 << bit_pos);
                }
                bit_index++;
            }
        }
    }

    return &packed_data;
  }



////////// con bits

using CombinedArray = std::array<uint64_t, 77>; // 77 elementos
PackedArray result = {}; // Inicializar con ceros

const PackedArray* concat_fen_legal_bits(const char *fen)
  {
    result = {};
    std::string fen_string(fen);
   // std::cerr << "fen:" << fen_string << std::endl;

    chess::Board board(fen_string); // Load FEN into board

    // Get board representation (13,8,8) and legal moves (64,8,8)
    auto fen_matrix = board_to_3D_bits_matrix(board);
    auto legal_moves = legal_moves_to_64_bitboards(board);

  

    CombinedArray combinedArray  = {}; // Inicializar con ceros

    // Copiar los 13 elementos del vector en las primeras posiciones
    std::copy(fen_matrix.begin(), fen_matrix.end(), combinedArray.begin());

    // Copiar los 64 elementos del array después del vector
    std::copy(legal_moves.begin(), legal_moves.end(), combinedArray.begin() + fen_matrix.size());

    for (size_t i = 0; i < combinedArray.size(); ++i) {
      uint64_t value = combinedArray[i];
      for (size_t j = 0; j < 8; ++j) {
          result[i * 8 + j] = static_cast<uint8_t>((value >> (j * 8)) & 0xFF);
      }
  }

    return &result;
  }

}