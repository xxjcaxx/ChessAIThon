import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PlayAiComponent } from './play-ai.component';

describe('PlayAiComponent', () => {
  let component: PlayAiComponent;
  let fixture: ComponentFixture<PlayAiComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PlayAiComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PlayAiComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
