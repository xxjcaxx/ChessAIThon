
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {Chess} from 'chess.js';


// Initialize the game instance correctly
const chess = new Chess();


const png2 = `[Event "Third Rosenwald Trophy"]
[Site "New York, NY USA"]
[Date "1956.10.17"]
[EventDate "1956.10.07"]
[Round "8"]
[Result "0-1"]
[White "Donald Byrne"]
[Black "Robert James Fischer"]
[ECO "D92"]
[WhiteElo "?"]
[BlackElo "?"]
[PlyCount "82"]

1. Nf3 Nf6 2. c4 g6 3. Nc3 Bg7 4. d4 O-O 5. Bf4 d5
6. Qb3 dxc4 7. Qxc4 c6 8. e4 Nbd7 9. Rd1 Nb6 10. Qc5 Bg4
11. Bg5 Na4 12. Qa3 Nxc3 13. bxc3 Nxe4 14. Bxe7 Qb6 15. Bc4 Nxc3
16. Bc5 Rfe8+ 17. Kf1 Be6 18. Bxb6 Bxc4+ 19. Kg1 Ne2+ 20. Kf1 Nxd4+
21. Kg1 Ne2+ 22. Kf1 Nc3+ 23. Kg1 axb6 24. Qb4 Ra4 25. Qxb6 Nxd1
26. h3 Rxa2 27. Kh2 Nxf2 28. Re1 Rxe1 29. Qd8+ Bf8 30. Nxe1 Bd5
31. Nf3 Ne4 32. Qb8 b5 33. h4 h5 34. Ne5 Kg7 35. Kg1 Bc5+
36. Kf1 Ng3+ 37. Ke1 Bb4+ 38. Kd1 Bb3+ 39. Kc1 Ne2+ 40. Kb1 Nc3+
41. Kc1 Rc2# 0-1`

chess.loadPgn(png2)
// -> true

// Source - https://stackoverflow.com/q/32685324
// Posted by user623990, modified by community. See post 'Timeline' for change history
// Retrieved 2026-09-26, License - CC BY-SA 3.0

function getMovesAsFENs(chessObj) {
    const moves = chessObj.history();
    const newGame = new Chess();
    const rows = [];
    for (const move of moves) {
      rows.push(`"${newGame.fen()}","${move}",0`);
      newGame.move(move);
    }
    return rows.join('\n');
}

const outputPath = fileURLToPath(new URL('./fens.csv', import.meta.url));
writeFileSync(outputPath, getMovesAsFENs(chess) + '\n');
console.log(`CSV generado: ${outputPath}`);


/**
 * 
 * 
 * 
 * 1Q6/5pk1/2p3p1/1p2N2p/1b5P/1bn5/2r3P1/2K5 w - - 16 42
[
  'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1',
  'rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2',
  'rnbqkb1r/pppppppp/5n2/8/2P5/5N2/PP1PPPPP/RNBQKB1R b KQkq - 0 2',
  'rnbqkb1r/pppppp1p/5np1/8/2P5/5N2/PP1PPPPP/RNBQKB1R w KQkq - 0 3',
  'rnbqkb1r/pppppp1p/5np1/8/2P5/2N2N2/PP1PPPPP/R1BQKB1R b KQkq - 1 3',
  'rnbqk2r/ppppppbp/5np1/8/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq - 2 4',
  'rnbqk2r/ppppppbp/5np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 4',
  'rnbq1rk1/ppppppbp/5np1/8/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQ - 1 5',
  'rnbq1rk1/ppppppbp/5np1/8/2PP1B2/2N2N2/PP2PPPP/R2QKB1R b KQ - 2 5',
  'rnbq1rk1/ppp1ppbp/5np1/3p4/2PP1B2/2N2N2/PP2PPPP/R2QKB1R w KQ - 0 6',
  'rnbq1rk1/ppp1ppbp/5np1/3p4/2PP1B2/1QN2N2/PP2PPPP/R3KB1R b KQ - 1 6',
  'rnbq1rk1/ppp1ppbp/5np1/8/2pP1B2/1QN2N2/PP2PPPP/R3KB1R w KQ - 0 7',
  'rnbq1rk1/ppp1ppbp/5np1/8/2QP1B2/2N2N2/PP2PPPP/R3KB1R b KQ - 0 7',
  'rnbq1rk1/pp2ppbp/2p2np1/8/2QP1B2/2N2N2/PP2PPPP/R3KB1R w KQ - 0 8',
  'rnbq1rk1/pp2ppbp/2p2np1/8/2QPPB2/2N2N2/PP3PPP/R3KB1R b KQ - 0 8',
  'r1bq1rk1/pp1nppbp/2p2np1/8/2QPPB2/2N2N2/PP3PPP/R3KB1R w KQ - 1 9',
  'r1bq1rk1/pp1nppbp/2p2np1/8/2QPPB2/2N2N2/PP3PPP/3RKB1R b K - 2 9',
  'r1bq1rk1/pp2ppbp/1np2np1/8/2QPPB2/2N2N2/PP3PPP/3RKB1R w K - 3 10',
  'r1bq1rk1/pp2ppbp/1np2np1/2Q5/3PPB2/2N2N2/PP3PPP/3RKB1R b K - 4 10',
  'r2q1rk1/pp2ppbp/1np2np1/2Q5/3PPBb1/2N2N2/PP3PPP/3RKB1R w K - 5 11',
  'r2q1rk1/pp2ppbp/1np2np1/2Q3B1/3PP1b1/2N2N2/PP3PPP/3RKB1R b K - 6 11',
  'r2q1rk1/pp2ppbp/2p2np1/2Q3B1/n2PP1b1/2N2N2/PP3PPP/3RKB1R w K - 7 12',
  'r2q1rk1/pp2ppbp/2p2np1/6B1/n2PP1b1/Q1N2N2/PP3PPP/3RKB1R b K - 8 12',
  'r2q1rk1/pp2ppbp/2p2np1/6B1/3PP1b1/Q1n2N2/PP3PPP/3RKB1R w K - 0 13',
  'r2q1rk1/pp2ppbp/2p2np1/6B1/3PP1b1/Q1P2N2/P4PPP/3RKB1R b K - 0 13',
  'r2q1rk1/pp2ppbp/2p3p1/6B1/3Pn1b1/Q1P2N2/P4PPP/3RKB1R w K - 0 14',
  'r2q1rk1/pp2Bpbp/2p3p1/8/3Pn1b1/Q1P2N2/P4PPP/3RKB1R b K - 0 14',
  'r4rk1/pp2Bpbp/1qp3p1/8/3Pn1b1/Q1P2N2/P4PPP/3RKB1R w K - 1 15',
  'r4rk1/pp2Bpbp/1qp3p1/8/2BPn1b1/Q1P2N2/P4PPP/3RK2R b K - 2 15',
  'r4rk1/pp2Bpbp/1qp3p1/8/2BP2b1/Q1n2N2/P4PPP/3RK2R w K - 0 16',
  'r4rk1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RK2R b K - 1 16',
  'r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3RK2R w K - 2 17',
  'r3r1k1/pp3pbp/1qp3p1/2B5/2BP2b1/Q1n2N2/P4PPP/3R1K1R b - - 3 17',
  'r3r1k1/pp3pbp/1qp1b1p1/2B5/2BP4/Q1n2N2/P4PPP/3R1K1R w - - 4 18',
  'r3r1k1/pp3pbp/1Bp1b1p1/8/2BP4/Q1n2N2/P4PPP/3R1K1R b - - 0 18',
  'r3r1k1/pp3pbp/1Bp3p1/8/2bP4/Q1n2N2/P4PPP/3R1K1R w - - 0 19',
  'r3r1k1/pp3pbp/1Bp3p1/8/2bP4/Q1n2N2/P4PPP/3R2KR b - - 1 19',
  'r3r1k1/pp3pbp/1Bp3p1/8/2bP4/Q4N2/P3nPPP/3R2KR w - - 2 20',
  'r3r1k1/pp3pbp/1Bp3p1/8/2bP4/Q4N2/P3nPPP/3R1K1R b - - 3 20',
  'r3r1k1/pp3pbp/1Bp3p1/8/2bn4/Q4N2/P4PPP/3R1K1R w - - 0 21',
  'r3r1k1/pp3pbp/1Bp3p1/8/2bn4/Q4N2/P4PPP/3R2KR b - - 1 21',
  'r3r1k1/pp3pbp/1Bp3p1/8/2b5/Q4N2/P3nPPP/3R2KR w - - 2 22',
  'r3r1k1/pp3pbp/1Bp3p1/8/2b5/Q4N2/P3nPPP/3R1K1R b - - 3 22',
  'r3r1k1/pp3pbp/1Bp3p1/8/2b5/Q1n2N2/P4PPP/3R1K1R w - - 4 23',
  'r3r1k1/pp3pbp/1Bp3p1/8/2b5/Q1n2N2/P4PPP/3R2KR b - - 5 23',
  'r3r1k1/1p3pbp/1pp3p1/8/2b5/Q1n2N2/P4PPP/3R2KR w - - 0 24',
  'r3r1k1/1p3pbp/1pp3p1/8/1Qb5/2n2N2/P4PPP/3R2KR b - - 1 24',
  '4r1k1/1p3pbp/1pp3p1/8/rQb5/2n2N2/P4PPP/3R2KR w - - 2 25',
  '4r1k1/1p3pbp/1Qp3p1/8/r1b5/2n2N2/P4PPP/3R2KR b - - 0 25',
  '4r1k1/1p3pbp/1Qp3p1/8/r1b5/5N2/P4PPP/3n2KR w - - 0 26',
  '4r1k1/1p3pbp/1Qp3p1/8/r1b5/5N1P/P4PP1/3n2KR b - - 0 26',
  '4r1k1/1p3pbp/1Qp3p1/8/2b5/5N1P/r4PP1/3n2KR w - - 0 27',
  '4r1k1/1p3pbp/1Qp3p1/8/2b5/5N1P/r4PPK/3n3R b - - 1 27',
  '4r1k1/1p3pbp/1Qp3p1/8/2b5/5N1P/r4nPK/7R w - - 0 28',
  '4r1k1/1p3pbp/1Qp3p1/8/2b5/5N1P/r4nPK/4R3 b - - 1 28',
  '6k1/1p3pbp/1Qp3p1/8/2b5/5N1P/r4nPK/4r3 w - - 0 29',
  '3Q2k1/1p3pbp/2p3p1/8/2b5/5N1P/r4nPK/4r3 b - - 1 29',
  '3Q1bk1/1p3p1p/2p3p1/8/2b5/5N1P/r4nPK/4r3 w - - 2 30',
  '3Q1bk1/1p3p1p/2p3p1/8/2b5/7P/r4nPK/4N3 b - - 0 30',
  '3Q1bk1/1p3p1p/2p3p1/3b4/8/7P/r4nPK/4N3 w - - 1 31',
  '3Q1bk1/1p3p1p/2p3p1/3b4/8/5N1P/r4nPK/8 b - - 2 31',
  '3Q1bk1/1p3p1p/2p3p1/3b4/4n3/5N1P/r5PK/8 w - - 3 32',
  '1Q3bk1/1p3p1p/2p3p1/3b4/4n3/5N1P/r5PK/8 b - - 4 32',
  '1Q3bk1/5p1p/2p3p1/1p1b4/4n3/5N1P/r5PK/8 w - - 0 33',
  '1Q3bk1/5p1p/2p3p1/1p1b4/4n2P/5N2/r5PK/8 b - - 0 33',
  '1Q3bk1/5p2/2p3p1/1p1b3p/4n2P/5N2/r5PK/8 w - - 0 34',
  '1Q3bk1/5p2/2p3p1/1p1bN2p/4n2P/8/r5PK/8 b - - 1 34',
  '1Q3b2/5pk1/2p3p1/1p1bN2p/4n2P/8/r5PK/8 w - - 2 35',
  '1Q3b2/5pk1/2p3p1/1p1bN2p/4n2P/8/r5P1/6K1 b - - 3 35',
  '1Q6/5pk1/2p3p1/1pbbN2p/4n2P/8/r5P1/6K1 w - - 4 36',
  '1Q6/5pk1/2p3p1/1pbbN2p/4n2P/8/r5P1/5K2 b - - 5 36',
  '1Q6/5pk1/2p3p1/1pbbN2p/7P/6n1/r5P1/5K2 w - - 6 37',
  '1Q6/5pk1/2p3p1/1pbbN2p/7P/6n1/r5P1/4K3 b - - 7 37',
  '1Q6/5pk1/2p3p1/1p1bN2p/1b5P/6n1/r5P1/4K3 w - - 8 38',
  '1Q6/5pk1/2p3p1/1p1bN2p/1b5P/6n1/r5P1/3K4 b - - 9 38',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1b4n1/r5P1/3K4 w - - 10 39',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1b4n1/r5P1/2K5 b - - 11 39',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1b6/r3n1P1/2K5 w - - 12 40',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1b6/r3n1P1/1K6 b - - 13 40',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1bn5/r5P1/1K6 w - - 14 41',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1bn5/r5P1/2K5 b - - 15 41',
  '1Q6/5pk1/2p3p1/1p2N2p/1b5P/1bn5/2r3P1/2K5 w - - 16 42'
]
 */