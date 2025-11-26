import {describe, expect, test } from "vitest";
import { GameState } from "../src/services/chessGameService";
import { BehaviorSubject } from "rxjs";

describe("GameState",()=>{
    describe("constructor",()=>{
        test("constructor", ()=>{
            const state = new GameState('r2qkbnr/ppp2ppp/2n5/1B2pQ2/4P3/8/PPP2PPP/RNB1K2R b KQkq - 3 7');
            expect(state).toBeInstanceOf(GameState);
            expect(state.state$).toBeInstanceOf(BehaviorSubject);
            const currentState = state.state$.getValue();
            expect(currentState.board[0][3]).toEqual({ type: 'q', square: 'd8', color: 'b' });
            expect(currentState.currentPlayer).toBe('b');
            expect(currentState.ended).toBe(false);
            expect(currentState.winner).toBe(null);
            expect(currentState.inCheck).toBe(false);
            expect(currentState.legalMoves).toBeInstanceOf(Array);
            //console.log(currentState.legalMoves);
            
        });
        test("set move",()=>{
            const state = new GameState('r2qkbnr/ppp2ppp/2n5/1B2pQ2/4P3/8/PPP2PPP/RNB1K2R b KQkq - 3 7');
            const currentState = state.state$.getValue();
            expect(currentState.board[0][1]).toBe(null);
            expect(currentState.board[0][0]).toEqual({ type: 'r', square: 'a8', color: 'b' });

            state.move = 'a8b8'; 
            const currentState2 = state.state$.getValue();
            expect(currentState2.board[0][0]).toBe(null);
            expect(currentState2.board[0][1]).toEqual({ type: 'r', square: 'b8', color: 'b' });
            expect(currentState2.currentPlayer).toBe('w'); 

            state.move = 'a8b8'; // invalid move no da error

        });
    });
});