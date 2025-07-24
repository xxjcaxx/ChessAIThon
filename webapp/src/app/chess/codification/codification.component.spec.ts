import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CodificationComponent } from './codification.component';

describe('CodificationComponent', () => {
  let component: CodificationComponent;
  let fixture: ComponentFixture<CodificationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CodificationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CodificationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
