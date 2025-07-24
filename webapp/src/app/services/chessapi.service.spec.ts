import { TestBed } from '@angular/core/testing';

import { ChessapiService } from './chessapi.service';

describe('ChessapiService', () => {
  let service: ChessapiService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ChessapiService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
