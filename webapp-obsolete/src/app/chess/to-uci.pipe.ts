import { Pipe, PipeTransform } from '@angular/core';
import { Chess } from 'chess.js'

@Pipe({
  name: 'toUci'
})
export class ToUciPipe implements PipeTransform {


  transform(value: string, fen: string): string {
    let chessCopy = new Chess(fen);
    let move = chessCopy.move(value);

    return  move.from ? `${move.from}${move.to}` : 'NaM';
  }

}
