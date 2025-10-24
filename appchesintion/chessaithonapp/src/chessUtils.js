export {chessPiecesUnicode,uciToMove,loadLocalStorage, decodeIdentificator}
import { Chess, validateFen } from 'chess.js'

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

const loadLocalStorage = () => {
  let bestMoves = [];
  const localStorageData = localStorage.getItem('best_moves');
  if (localStorageData) {
    try {
      bestMoves = JSON.parse(localStorageData);
    }
    catch (e) {
    }
  }
  return bestMoves;
}


const decodeIdentificator = (identificator) => {
    console.log(identificator);
    if (identificator) {
      let fen = decodeURIComponent(identificator);
      const fenArray = fen.split('');
      if(fenArray.at(-1)==='0'){
        fenArray[fenArray.length-1]='1'
        fen = fenArray.join('');
      }

      if (validateFen(fen).ok) {
        return fen;
      }
      else{
        console.log(validateFen(fen).error);
        
        return null;
      }
    }
    return null;
  }