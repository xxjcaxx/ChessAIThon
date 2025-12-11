export {chessPiecesUnicode,uciToMove,loadLocalStorage, decodeIdentificator, generateMiniChessBoard}
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

const generateMiniChessBoard = (board) => {

  function renderBoardToCanvas(boardData, size = 150) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');

    if (!ctx) return canvas;

    const squareSize = size / 8;
    const lightSquareColor = '#f0d9b5';
    const darkSquareColor = '#b58863';

    // Función auxiliar para dibujar un cuadrado
    const drawSquare = (r, c, color) => {
        ctx.fillStyle = color;
        ctx.fillRect(c * squareSize, r * squareSize, squareSize, squareSize);
    };

    // Función auxiliar para dibujar una pieza
    const drawPiece = (r, c, piece) => {
        // Configuración para las piezas Unicode
        ctx.font = `${squareSize * 0.75}px Arial`; 
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Determinar color de la pieza basado en Unicode (simple)
        const isWhite = piece === '♔' || piece === '♕' || piece === '♖' || piece === '♗' || piece === '♘' || piece === '♙';
        ctx.fillStyle = isWhite ? '#FFFFFF' : '#000000';
        
        ctx.fillText(piece, c * squareSize + squareSize / 2, r * squareSize + squareSize / 2);
    };

    // Recorrido y Dibujo
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            // 1. Dibujar el cuadrado
            const color = (r + c) % 2 === 0 ? lightSquareColor : darkSquareColor;
            drawSquare(r, c, color);

            // 2. Dibujar la pieza
            const piece = boardData[r][c];
            if (piece !== 0) {
                drawPiece(r, c, piece);
            }
        }
    }

    return canvas;
}

    const miniBoard = board.map(row => row.map(cell => {
        if (cell === null) return 0;
        const pieceChar = cell.type.toUpperCase();
        const isWhite = cell.color === 'w';
        const unicodePiece = chessPiecesUnicode[isWhite ? pieceChar : pieceChar.toLowerCase()];
        return unicodePiece;
    }));
    //console.log(miniBoard);
    const canvas = renderBoardToCanvas(miniBoard);
    
    return canvas.toDataURL('image/png');
}