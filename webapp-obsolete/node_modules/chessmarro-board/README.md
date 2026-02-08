# Chessmarro Board

> This library is part of a educational chess project. It will not be maintained and there are better options like Chessboard2.

It tries to be updated with the state of the art in 2025 of library creation for javascript vanilla and typescript and to be usable in frameworks.

## Usage

Install:

```bash
npm install chessmarro-board
```

It declares a Web Component and you can use it in many ways. You can see the examples folder. Basically, you have to import the library and initialize:

```javascript
import { initChessmarroBoard } from "chessmarro-board";

initChessmarroBoard();
```

The function `initChessmarroBoard` define the tag to be used in your project. Then, you can add it in your HTML:

```html
<chessmarro-board
  data-fen="3b2k1/1p3p2/p1p5/2P4p/1P2P1p1/5p2/5P2/4RK2 w - - 0 0"
  data-black-color="#400"
  data-white-color="#e77"
  data-numbers="true"
  data-numbers-color="#555"
></chessmarro-board>
```

Or just create one by Javascript:

```javascript
import { initChessmarroBoard, setStartPosition } from "chessmarro-board";

document.addEventListener("DOMContentLoaded", () => {
  const board = setStartPosition();
  initChessmarroBoard();

  const divApp = document.querySelector("#javascript");
  const boardComponent = document.createElement("chessmarro-board");
  boardComponent.board = board;
  divApp.append(boardComponent);
});
```

As you can see in the first example, you can add to `data-` some settings. `data-fen` or `boardComponent.dataset.fen` is to set a start fen position, `data-black-color` and `data-white-color` to set the colors of the board, `data-numbers` if you want to see the numbers and `data-numbers-color` the color of data numbers background.

You can interact with the board, if you drag and drop a piece it will trigger an event called `'chessmarro-move'` with some `detail` of the move:

```json
{
  "origin": {
    "piece": "p",
    "position": "f7",
    "matrix_position": "[5,1]"
  },
  "destiny": {
    "position": "d3",
    "matrix_position": "[3,5]"
  },
  "uci": "f7d3"
}
```

You can set and get the FEN position with `setFen`, `getFen`.

## CSS

The Board is inside a Shadow Root, so you have to consider it if you want to modify styles.

The board tries to fill the element at 100% and to be responsive at any size.

You can change colors by `data-`.

The base of the package is https://github.com/kbysiec/vite-vanilla-ts-lib-starter
