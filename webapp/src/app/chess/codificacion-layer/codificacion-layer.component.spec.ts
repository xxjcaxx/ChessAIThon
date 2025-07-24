import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CodificacionLayerComponent } from './codificacion-layer.component';

describe('CodificacionLayerComponent', () => {
  let component: CodificacionLayerComponent;
  let fixture: ComponentFixture<CodificacionLayerComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CodificacionLayerComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CodificacionLayerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
