import { Injectable } from '@angular/core';
import { Chess } from 'chess.js'

const types = {
  'p': 0, // Pawns layer 0 and 1
  'n': 1,
  'b': 2,
  'r': 3,
  'q': 4,
  'k': 5
}

const letters: { [key: string]: number } = {
  a: 1,
  b: 2,
  c: 3,
  d: 4,
  e: 5,
  f: 6,
  g: 7,
  h: 8,
  1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 6,
  7: 7,
  8: 8
}



const codes: { [key: string]: number } = {};
let i = 13;

// Todas las 56 jugadas regulares
for (let nSquares = 1; nSquares < 8; nSquares++) {
  for (const [dx, dy] of [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]) {
    codes[`${nSquares * dx},${nSquares * dy}`] = i;
    i++;
  }
}

// 8 movimientos del caballo
const knightMoves = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
for (const [dx, dy] of knightMoves) {
  codes[`${dx},${dy}`] = i;
  i++;
}


@Injectable({
  providedIn: 'root'
})
export class CodificationService {

  chess: Chess;

  name_layers = ["white pawns", "black pawns", "white knights", "black knights", "white bishops", "black bishops",
    "white rooks", "black rooks", "white queen", "black queen", "white king", "black king", "**Turn**",
    "1 North moves", "1 NE moves", "1 East moves", "1 SE moves", "1 South moves", "1 SW moves", "1 West moves", "1 NW moves",
    "2 North moves", "2 NE moves", "2 East moves", "2 SE moves", "2 South moves", "2 SW moves", "2 West moves", "2 NW moves",
    "3 North moves", "3 NE moves", "3 East moves", "3 SE moves", "3 South moves", "3 SW moves", "3 West moves", "3 NW moves",
    "4 North moves", "4 NE moves", "4 East moves", "4 SE moves", "4 South moves", "4 SW moves", "4 West moves", "4 NW moves",
    "5 North moves", "5 NE moves", "5 East moves", "5 SE moves", "5 South moves", "5 SW moves", "5 West moves", "5 NW moves",
    "6 North moves", "6 NE moves", "6 East moves", "6 SE moves", "6 South moves", "6 SW moves", "6 West moves", "6 NW moves",
    "7 North moves", "7 NE moves", "7 East moves", "7 SE moves", "7 South moves", "7 SW moves", "7 West moves", "7 NW moves",
    "E2N Knight", "2EN Knight", "2ES Knight", "E2S Knight", "W2S Knight", "2WS Knight", "2WN Knight", "W2N Knight",
    "none", "none", "none", "none", "none", "none", "none", "none",
    "none", "none", "none", "none", "none", "none", "none", "none",
  ];

  constructor() {
    this.chess = new Chess();
  }

  concat_fen_legal(fen: string): number[][][] {

    this.chess.load(fen);

    const layers = Array(77).fill(0).map(row => Array(8).fill(0).map(row => Array(8).fill(0)));
    this.chess.board().forEach((row, i) => row.forEach((col, j) => {
      if (col) {
        //console.log(col,types[col.type]*2+ (col.color === 'b' ? 1 : 0));

        layers[types[col.type] * 2 + (col.color === 'b' ? 1 : 0)][i][j] = 1;
      }
    }));

    this.chess.moves({ verbose: true }).forEach(m => {
      let [fromx, fromy, tox, toy] = m.lan.split('').map(x => letters[x]);
      let mx = tox - fromx;
      let my = toy - fromy;
      let layer = codes[`${mx},${my}`];
      //console.log(m, fromx, fromy, tox, toy, mx, my, layer, this.name_layers[layer]);

      layers[layer][8 - fromy][fromx - 1] = m.piece;
      layers[layer][8 - toy][tox - 1] = '*';
    });


    return layers;

  }

}
