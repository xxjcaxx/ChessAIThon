class HomeComponent extends HTMLElement {

    connectedCallback() {
        this.innerHTML = `
<div class="main-content">
  <section class="hero is-fullheight-with-navbar">
    <div class="hero-body">
      <div class="container">
        <div class="box home-hero-box">
          <div class="columns is-vcentered is-variable is-6">
            <div class="column is-narrow has-text-centered">
              <figure class="image is-128x128 is-inline-block">
                <img src="/logoblanc.png" alt="Logo de Chess Minds" class="home-logo">
              </figure>
            </div>
            <div class="column has-text-centered-mobile has-text-left-tablet">
              <h1 class="title has-text-white is-2 mb-3">Chess Minds</h1>
              <p class="subtitle has-text-white mb-1">Project ChessAIthon</p>
              <p class="has-text-white">
                Explore Chess AI.
                <br>
                This web application provides the visual tool to explore how the AI for chess designed for this project works. Here you can
                play scenarios and store your moves. You can play with your AI to improve it.
              </p>
            </div>
          </div>
        </div>

        <div class="columns is-multiline is-centered mt-5">
          <div class="column is-6-tablet is-4-desktop">
            <a href="#scenarios" class="home-card-link">
              <div class="card home-nav-card">
                <div class="card-content">
                  <figure class="image is-16by9 home-card-image mb-3">
                    <img src="/scenarios.png" alt="Placeholder image for Chess Scenarios">
                  </figure>
                  <p class="title is-5 home-card-title mb-2">Chess Scenarios</p>
                  <p class="home-card-text">
                    Play chess scenarios and manage your best moves for each one.
                    <br>
                    Export your scenarios and best moves.
                  </p>
                </div>
              </div>
            </a>
          </div>

          <div class="column is-6-tablet is-4-desktop">
            <a href="#play" class="home-card-link">
              <div class="card home-nav-card">
                <div class="card-content">
                  <figure class="image is-16by9 home-card-image mb-3">
                    <img src="/play.png" alt="Placeholder image for Play Chess">
                  </figure>
                  <p class="title is-5 home-card-title mb-2">Play Chess</p>
                  <p class="home-card-text">Here you can play chess against your AI. You can also see how two AI play one against the other.</p>
                </div>
              </div>
            </a>
          </div>

          <div class="column is-6-tablet is-4-desktop">
            <a href="#representation" class="home-card-link">
              <div class="card home-nav-card">
                <div class="card-content">
                  <figure class="image is-16by9 home-card-image mb-3">
                    <img src="/representation.png" alt="Placeholder image for Chess Representation">
                  </figure>
                  <p class="title is-5 home-card-title mb-2">Chess Representation</p>
                  <p class="home-card-text">See how a FEN is represented in 77x8x8 Layers format from our project.</p>
                </div>
              </div>
            </a>
          </div>

          <div class="column is-6-tablet is-4-desktop">
            <a href="#ai" class="home-card-link">
              <div class="card home-nav-card">
                <div class="card-content">
                  <figure class="image is-16by9 home-card-image mb-3">
                    <img src="/ai.png" alt="Placeholder image for Chess AI">
                  </figure>
                  <p class="title is-5 home-card-title mb-2">Chess AI</p>
                  <p class="home-card-text">See how your AI can decide the best move</p>
                </div>
              </div>
            </a>
          </div>

          <div class="column is-6-tablet is-4-desktop">
            <a href="https://chessaithon.pixel-online.org/" class="home-card-link">
              <div class="card home-nav-card">
                <div class="card-content">
                  <figure class="image is-16by9 home-card-image mb-3">
                    <img src="/logobanderasancho.png" alt="Placeholder image for About section">
                  </figure>
                  <p class="title is-5 home-card-title mb-2">About</p>
                  <p class="home-card-text">Visit the official web page of the project ChessAIthon</p>
                </div>
              </div>
            </a>
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