import { Component } from '@angular/core';
import {MatGridListModule} from '@angular/material/grid-list'; 

@Component({
    selector: 'app-upload',
    imports: [MatGridListModule],
    templateUrl: './upload.component.html',
    styleUrl: './upload.component.css'
})
export class UploadComponent {
  selectedFile: File | null = null;
  onFileSelected(event: any): void {
    const fileInput = event.target as HTMLInputElement;
    if (fileInput.files && fileInput.files.length > 0) {
      this.selectedFile = fileInput.files[0];
    } else {
      this.selectedFile = null;
    }
  }
  onSubmit(): void {
    // Aquí puedes implementar la lógica para subir el archivo.
    // Puedes usar un servicio para manejar la lógica de subida de archivos.
    console.log('Archivo seleccionado:', this.selectedFile);
  }
}
