import { AfterViewInit, Component, OnInit, QueryList, ViewChild, ViewChildren } from '@angular/core';
import { BoardComponent } from '../board/board.component';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatChipListbox, MatChipOption, MatChipsModule } from '@angular/material/chips';
import { Chess } from 'chess.js'
import { Scenario } from '../scenario';
import { NgClass, NgStyle } from '@angular/common';
import { BehaviorSubject, debounceTime } from 'rxjs';
import { ChessapiService } from '../../services/chessapi.service';

@Component({
    selector: 'app-boards',
    imports: [BoardComponent, MatGridListModule, MatChipsModule ],
    templateUrl: './boards.component.html',
    styleUrl: './boards.component.css'
})
export class BoardsComponent implements OnInit, AfterViewInit {

  constructor (private chessapiservice: ChessapiService){}

  chess = new Chess();
  turn = '';
  scenario: Scenario = { fen: '6qn/p5kp/PpQ2rp1/2n1NbP1/1PP2P1P/B1P1P3/8/7K w - - 1 44', legal_moves_uci: [], best: 'c6f6' };
  result: Scenario = { fen: '6qn/p5kp/PpQ2rp1/2n1NbP1/1PP2P1P/B1P1P3/8/7K w - - 1 44', legal_moves_uci: [], best: 'c6f6' };
  possibleMoves: string[] = [];
  availablePieces: string[] = [];
  movesSubject: BehaviorSubject<string> = new BehaviorSubject('reset');

  @ViewChild('availablePiecesDiv') apd! : MatChipListbox;
  @ViewChild('possibleMovesDiv') pmd! : MatChipListbox;

  resetBoards(){
    this.chess.load(this.scenario.fen);
    this.turn = this.chess.turn();
    this.availablePieces = ['B','K','N','P','Q','R'].map(type => `${this.turn}${type}`);
    //this.possibleMoves = this.chess.moves();
  }

  ngOnInit(): void {
    this.resetBoards();
    this.movesSubject.pipe(debounceTime(100)).subscribe(move => {
      this.resetBoards();
      if(move !== 'reset') this.chess.move(move);
      this.result.fen = this.chess.fen();
    });
    this.chessapiservice.getRandomScenario();
  }

  ngAfterViewInit(): void {
    this.possibleMoves = this.chess.moves();
    this.apd.chipSelectionChanges.subscribe(
      c=> {
       this.resetBoards();
       [this.pmd.selected].flat().filter(m => m).forEach(s => s.deselect() );
        let types = [this.apd.selected].flat().map(s => s.value);
        if(types.length > 0){
          this.possibleMoves = types.map(v=> this.chess.moves({piece: v.split('')[1].toLowerCase()})).flat()
        }
        else {
          this.possibleMoves = this.chess.moves();
        }


      })
  }

  showMove(move: string) {
    this.movesSubject.next(move);
  }

  resetMove(event: Event) {
    let selected = [this.pmd.selected].flat().filter(m => m);
    if(selected.length > 0){
      this.movesSubject.next(selected[0].value);
    }else {
      this.movesSubject.next('reset');
    }

  }



}
