# webapp

Angular web application for ChessAIThon.

Overview

- Built with Angular (see `package.json` in this folder).
- Uses Angular libraries and several chess-related packages (`chess.js`, `chessmarro-board`, `@chrisoakman/chessboard2`).

Local development

```bash
cd webapp
npm install
npm run start
```

Common scripts (from package.json)

- `npm run start` — runs `ng serve` (development server).
- `npm run build` — builds the production bundle.
- `npm run test` — runs unit tests.

Notes

- The Angular CLI is used; if `ng` is not available globally, `npm run start` uses the local CLI.
- App source is in `src/` and static assets are in `public/`.
