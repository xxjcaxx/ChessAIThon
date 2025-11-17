import { Component } from '@angular/core';
import {MatToolbarModule} from '@angular/material/toolbar';
//import {sum} from "chessmarro-board"

@Component({
    selector: 'app-header',
    imports: [MatToolbarModule],
    templateUrl: './header.component.html',
    styleUrl: './header.component.css'
})
export class HeaderComponent {

}


//console.log(sum(2, 3)); // 5
