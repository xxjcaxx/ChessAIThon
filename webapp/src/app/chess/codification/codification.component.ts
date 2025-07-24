import { Component, Input, OnInit } from '@angular/core';
import { Chess } from 'chess.js';
import { Scenario } from '../scenario';
import { BoardComponent } from '../board/board.component';

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
import { MatGridListModule } from '@angular/material/grid-list';
import { MatChipListbox, MatChipOption, MatChipsModule } from '@angular/material/chips';
import { CodificacionLayerComponent } from '../codificacion-layer/codificacion-layer.component';
import { CodificationService } from '../codification.service';

@Component({
  selector: 'app-codification',
  imports: [BoardComponent, MatGridListModule, MatChipsModule, MatCard, MatCardContent,
    CodificacionLayerComponent,
    ReactiveFormsModule, MatFormField, MatLabel, MatSlideToggle, MatInputModule, MatIconModule, ToUciPipe],
  templateUrl: './codification.component.html',
  styleUrl: './codification.component.css'
})
export class CodificationComponent implements OnInit{

  @Input('fen') fenParameter?: string;

  chess: Chess;
    scenario: Scenario = { fen: '6qn/p5kp/PpQ2rp1/2n1NbP1/1PP2P1P/B1P1P3/8/7K w - - 1 44', legal_moves_uci: [], best: 'c6f6' };
    currentFen: string = '6qn/p5kp/PpQ2rp1/2n1NbP1/1PP2P1P/B1P1P3/8/7K w - - 1 44'
    turn = '';
    possibleMoves: string[] = [];
    availablePieces: string[] = [];
    fenForm: FormGroup;

    layers: number[][][] = Array(77).fill(0).map(row => Array(8).fill(0).map(row => Array(8).fill(0)));
    name_layers: string[] = []

    constructor(private fb: FormBuilder, private router: Router, private condificationService: CodificationService){
      this.chess = new Chess;
      this.fenForm = this.fb.group(
        {
          fen: [''],
        }
      );
      this.name_layers = condificationService.name_layers;
    }

    setFen(){
      this.router.navigate([], {
        queryParams: { fen: this.fenForm.get('fen')?.value },
        queryParamsHandling: 'merge' // Mantiene otros parámetros en la URL
      });
      this.scenario.fen = this.fenForm.get('fen')?.value;
      this.currentFen = this.fenForm.get('fen')?.value;

      this.layers = this.condificationService.concat_fen_legal( this.fenForm.get('fen')?.value);

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
      this.layers = this.condificationService.concat_fen_legal(this.currentFen);
    }

}
