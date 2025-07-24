import { bootstrapApplication } from '@angular/platform-browser';
import { appConfig } from './app/app.config';
import { AppComponent } from './app/app.component';

import {initChessmarroBoard} from "chessmarro-board"

if (!customElements.get('chessmarro-board')) {
  initChessmarroBoard();
}

bootstrapApplication(AppComponent, appConfig)
  .catch((err) => console.error(err));
