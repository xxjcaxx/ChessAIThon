import { AfterViewInit, Component, Input, CUSTOM_ELEMENTS_SCHEMA } from '@angular/core';
import { Scenario } from '../scenario';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { JsonPipe } from '@angular/common';
import { MatGridListModule } from '@angular/material/grid-list';
import { BehaviorSubject } from 'rxjs';



@Component({
    selector: 'app-board',
    imports: [
        MatSlideToggleModule,
        MatGridListModule
    ],
    templateUrl: './board.component.html',
    styleUrl: './board.component.css',
    schemas: [CUSTOM_ELEMENTS_SCHEMA], // Permite Web Components
})
export class BoardComponent implements AfterViewInit {

  @Input() name: string = 'board';


  chessboard: {} | null = null;


  @Input() set board(value: string){
    this.fenSubject.next(value);
  }
  fenSubject: BehaviorSubject<string> = new BehaviorSubject('start');
  constructor(){
    //this.chessboard = this.initBoard();
  }

  ngAfterViewInit(): void {

    setTimeout(()=>{  // Settimeout waits main thread
      this.chessboard = this.initBoard(this.fenSubject.value);
          // @ts-ignore
    this.fenSubject.subscribe(fen => this.chessboard.setPosition(fen, 'super slow'));
    },0)


  }

  initBoard(fen: string) {
    let that = this;
    const config = {
      showNotation: true,
      sparePieces: true,
      position: fen,
      draggable: true,
      dropOffBoard: 'snapback', // this is the default
      onMoveEnd(oldPos: any, newPos: any) {
        console.log(oldPos, newPos);
      },
      // @ts-ignore
      onDrop(source, target, piece, newPos, oldPos, orientation) {
        //console.log({ source, target, piece, newPos, oldPos, orientation });
        // @ts-ignore
        console.log(that.chessboard.getPosition('fen'));


      }
    }
    // @ts-ignore
    return Chessboard2(this.name, config)
  }

}
