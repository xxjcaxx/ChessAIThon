export {chessPiecesUnicode,uciToMove}

const chessPiecesUnicode = {
  'P': '♙', // Peón blanco
  'N': '♘', // Caballo blanco
  'B': '♗', // Alfil blanco
  'R': '♖', // Torre blanca
  'Q': '♕', // Reina blanca
  'K': '♔', // Rey blanco
  'p': '♟', // Peón negro
  'n': '♞', // Caballo negro
  'b': '♝', // Alfil negro
  'r': '♜', // Torre negra
  'q': '♛', // Reina negra
  'k': '♚'  // Rey negro
};

const uciToMove = (uci) => {
  const letters = [" ", "a", "b", "c", "d", "e", "f", "g", "h", " "];
  const [oy, ox, dy, dx] = uci.split("");
  return [
    letters.indexOf(oy) - 1,
    8 - parseInt(ox),
    letters.indexOf(dy) - 1,
    8 - parseInt(dx),
  ];
};