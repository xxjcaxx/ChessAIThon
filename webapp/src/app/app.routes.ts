import { Routes } from '@angular/router';
import { BoardComponent } from './chess/board/board.component';
import { UploadComponent } from './chess/upload/upload.component';
import { BoardsComponent } from './chess/boards/boards.component';
import { TreeVisualizerComponent } from './mcts/tree-visualizer/tree-visualizer.component';
import { HomeComponent } from './components/home/home.component';
import { PlayAiComponent } from './chess/play-ai/play-ai.component';
import { CodificationComponent } from './chess/codification/codification.component';

export const routes: Routes = [
  {path: 'home', component: HomeComponent},
    {path: 'random', component: BoardsComponent},
   {path: 'upload',  component: UploadComponent},
   {path: 'play_ai',  component: PlayAiComponent},
   {path: 'codification',  component: CodificationComponent},
   {path: 't', component: TreeVisualizerComponent},
   {path: '**', pathMatch: 'full', redirectTo: 'random'}

];
