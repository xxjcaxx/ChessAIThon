import { Component, Input } from '@angular/core';
import {NgClass} from '@angular/common';


@Component({
  selector: 'app-codificacion-layer',
  imports: [NgClass],
  templateUrl: './codificacion-layer.component.html',
  styleUrl: './codificacion-layer.component.css'
})
export class CodificacionLayerComponent {

  @Input() layer: number[][] = Array(8).fill(0).map(row => Array(8).fill(0));
  @Input() name: string = 'layer';

  constructor(){

  }


}
