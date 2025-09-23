class HomeComponent extends HTMLElement {

    connectedCallback() {
        this.innerHTML = `
<div class="main-content">
    <section class="hero is-fullheight-with-navbar">
      <div class="hero-body has-text-centered">
        <div class="container">
          <h1 class="title has-text-white is-2">ChessAIthon</h1>
          <p class="subtitle has-text-white">Explore Chess AI</p>
          <div class="columns is-multiline is-centered mt-6">
            <!-- Cards -->
            <div class="column is-4-desktop is-6-tablet">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Inicio</p>
                </div>
              </div>
            </div>
            <div class="column is-4-desktop is-6-tablet">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Acerca de</p>
                </div>
              </div>
            </div>
            <div class="column is-4-desktop is-6-tablet">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Servicios</p>
                </div>
              </div>
            </div>
            <div class="column is-4-desktop is-6-tablet">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Proyectos</p>
                </div>
              </div>
            </div>
            <div class="column is-4-desktop is-6-tablet">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Blog</p>
                </div>
              </div>
            </div>
            <div class="column is-4-desktop is-6-tablet">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Contacto</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
`;
    }


}

customElements.define("chess-home", HomeComponent);