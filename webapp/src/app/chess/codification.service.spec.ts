import { TestBed } from '@angular/core/testing';

import { CodificationService } from './codification.service';

describe('CodificationService', () => {
  let service: CodificationService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CodificationService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
