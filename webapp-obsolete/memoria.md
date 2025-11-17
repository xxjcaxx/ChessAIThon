npm install @chrisoakman/chessboard2


"styles": [
  "src/styles.scss",
  "node_modules/@chrisoakman/chessboard2/dist/chessboard2.min.css"
],
"scripts": [
  "node_modules/@chrisoakman/chessboard2/dist/chessboard2.min.js"
],


  initBoard(){
      // @ts-ignore
   this.board = Chessboard2('board1','start')
  }
