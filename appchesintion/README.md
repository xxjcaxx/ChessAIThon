# appchesintion

Frontend/demo directory containing the small Vite-based app used as a demo for ChessAIThon.

Subproject: `chessaithonapp`

See `chessaithonapp/` for the actual web app.

Quick start

1. cd into the app folder:

```bash
cd appchesintion/chessaithonapp
```

2. Install dependencies and start the dev server:

```bash
npm install
npm run dev
```

Build

```bash
npm run build
npm run preview
```

Notes

- The app uses Vite and depends on libraries such as `chess.js`, `chessmarro-board` and `rxjs`.
- Static data used by the app is under `public/` (for example `chess_endgames.csv`).
