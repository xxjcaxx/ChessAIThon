import { AfterViewInit, Component, Input, OnInit, QueryList, ViewChild, ViewChildren } from '@angular/core';
import { BoardComponent } from '../board/board.component';
import { MatGridListModule } from '@angular/material/grid-list';
import { MatChipListbox, MatChipOption, MatChipsModule } from '@angular/material/chips';
import { Chess } from 'chess.js'
import { Scenario } from '../scenario';
import { NgClass, NgStyle } from '@angular/common';
import { BehaviorSubject, debounceTime } from 'rxjs';
import { ChessapiService } from '../../services/chessapi.service';
import { FormBuilder, FormGroup } from '@angular/forms';
import { MatCard, MatCardContent } from '@angular/material/card';
import { ReactiveFormsModule } from '@angular/forms';
import { MatFormField, MatLabel } from '@angular/material/form-field';
import { MatSlideToggle } from '@angular/material/slide-toggle';
import { MatInputModule } from '@angular/material/input';
import { MatIconButton } from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import { ActivatedRoute, Router } from '@angular/router';
import { ToUciPipe } from '../to-uci.pipe';

@Component({
  selector: 'app-play-ai',
  imports: [BoardComponent, MatGridListModule, MatChipsModule, MatCard, MatCardContent, ReactiveFormsModule, MatFormField, MatLabel, MatSlideToggle, MatInputModule, MatIconModule, ToUciPipe],
  templateUrl: './play-ai.component.html',
  styleUrl: './play-ai.component.css'
})
export class PlayAiComponent implements OnInit, AfterViewInit{


  @Input('fen') fenParameter?: string;


  chess: Chess;
  scenario: Scenario = { fen: '6qn/p5kp/PpQ2rp1/2n1NbP1/1PP2P1P/B1P1P3/8/7K w - - 1 44', legal_moves_uci: [], best: 'c6f6' };
  currentFen: string = '6qn/p5kp/PpQ2rp1/2n1NbP1/1PP2P1P/B1P1P3/8/7K w - - 1 44'
  turn = '';
  possibleMoves: string[] = [];
  availablePieces: string[] = [];
  movesSubject: BehaviorSubject<string> = new BehaviorSubject('reset');

  @ViewChild('availablePiecesDiv') apd! : MatChipListbox;
  @ViewChild('possibleMovesDiv') pmd! : MatChipListbox;


  serversForm: FormGroup;
  fenForm: FormGroup;
  switchStateW = true; // Inicialmente activado
  switchStateB = true; // Inicialmente activado

  constructor(private fb: FormBuilder, private router: Router) {
    this.chess = new Chess;
    this.serversForm = this.fb.group({
      serverW: [''],
      serverB: [''],
      switchW: [true],
      switchB: [true],
    });
    this.fenForm = this.fb.group(
      {
        fen: [''],
      }
    );
  }

  ngOnInit(): void {
    if(this.fenParameter){
      this.scenario.fen = this.fenParameter;
      this.currentFen = this.fenParameter;
    }
    else{
      this.chess.reset()
      this.scenario.fen = this.chess.fen();
      this.currentFen =  this.scenario.fen;
    }
    this.resetBoards();
    this.movesSubject.pipe(debounceTime(100)).subscribe(move => {
      this.resetBoards();
      if(move !== 'reset') this.chess.move(move);
      this.scenario.fen = this.chess.fen();
    });

  }

  ngAfterViewInit(): void {
    this.possibleMoves = this.chess.moves();
    this.apd.chipSelectionChanges.subscribe(
      c=> {
       this.resetBoards();
       [this.pmd.selected].flat().filter(m => m).forEach(s => s.deselect() );
      this.getAvailablePieces();


      })
  }

  getAvailablePieces(){
    let types = this.apd ? [this.apd.selected].flat().map(s => s.value) : [];
    if(types.length > 0){
      this.possibleMoves = types.map(v=> this.chess.moves({piece: v.split('')[1].toLowerCase()})).flat()
    }
    else {
      this.possibleMoves = this.chess.moves();
    }
  }


  showMove(move: string) {
    this.movesSubject.next(move);
  }

  resetBoards(){
    this.chess.load(this.currentFen);
    this.turn = this.chess.turn();
    this.availablePieces = ['B','K','N','P','Q','R'].map(type => `${this.turn}${type}`);
    this.getAvailablePieces();
  }

  resetMove(event: Event) {
    let selected = [this.pmd.selected].flat().filter(m => m);
    if(selected.length > 0){
      this.movesSubject.next(selected[0].value);
    }else {
      this.movesSubject.next('reset');
    }

  }

  toggleInputW(state: boolean) {
    this.switchStateW = state;
    if (state) {
      this.serversForm.get('serverW')?.enable();
    } else {
      this.serversForm.get('serverW')?.disable();
    }
  }
  toggleInputB(state: boolean) {
    this.switchStateB = state;
    if (state) {
      this.serversForm.get('serverB')?.enable();
    } else {
      this.serversForm.get('serverB')?.disable();
    }
  }

  setFen(){
    this.router.navigate([], {
      queryParams: { fen: this.fenForm.get('fen')?.value },
      queryParamsHandling: 'merge' // Mantiene otros parámetros en la URL
    });
    this.scenario.fen = this.fenForm.get('fen')?.value;
    this.currentFen = this.fenForm.get('fen')?.value;
  }
}
