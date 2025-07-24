import { Component } from '@angular/core';
import { BoardComponent } from '../../chess/board/board.component';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import {MatListModule} from '@angular/material/list';
import {MatGridListModule} from '@angular/material/grid-list';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
    selector: 'app-main',
    imports: [
        MatSidenavModule,
        MatButtonModule,
        MatDividerModule,
        MatListModule,
        MatGridListModule,
        RouterLink,
        RouterOutlet,
        RouterLinkActive
    ],
    templateUrl: './main.component.html',
    styleUrl: './main.component.css'
})
export class MainComponent {

}
