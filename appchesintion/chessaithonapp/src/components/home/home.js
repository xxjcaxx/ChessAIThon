class HomeComponent extends HTMLElement {

    connectedCallback() {
        this.innerHTML = `
<div class="main-content">
    <section class="hero is-fullheight-with-navbar">
      <div class="hero-body has-text-centered">
        <div class="container">


<div class="container mt-6">
  <div class="is-flex is-align-items-stretch is-justify-content-center">
    <figure class="mr-4" style="height: 9em;">
  <img src="/logoblanc.png" alt="Logo de Chess Minds" style="height: 100%; object-fit: contain;">
</figure>

    <div style="max-width: 500px; text-align: center;">
      <h1 class="title has-text-white is-2 mb-3">Chess Minds</h1>
      <p class="subtitle has-text-white mb-1">Project ChessAIthon</p>
      <p class="has-text-white">Explore Chess AI.
      <br>
      This web application provides the visual tool to explore how the AI for chess designed for this project works. Here you can 
      play scenarios and store your moves. You can play with your AI to improve it. 
      </p>
    </div>
  </div>
</div>



          <div class="columns is-multiline is-centered mt-6">
            <!-- Cards -->

            <div class="column is-4-desktop is-6-tablet">
            <a href="#scenarios">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Chess Scenarios</p>
                  <p class="">Play chess scenarios and manage your best moves for each one. 
                  <br>
                  Export your scenarios and best moves.
                  </p>
                </div>
              </div>
              </a>
            </div>
            
            <div class="column is-4-desktop is-6-tablet">
            <a href="#play">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Play Chess</p>
                  <p class="">Here you can play chess against your AI. You can also see how two AI play one against the other. 
                  </p>
                </div>
              </div>
              </a>
            </div>
            

            <div class="column is-4-desktop is-6-tablet">
             <a href="#representation">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Chess Representation</p>
                  <p class="">See how a FEN is represented in 77x8x8 Layers format from our project.  
                  </p>
                </div>
              </div>
              </a>
            </div>
            

            <div class="column is-4-desktop is-6-tablet">
             <a href="#ai">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">Chess AI</p>
                  <p class="">See how your AI can decide the best move 
                  </p>
                </div>
              </div>
              </a>
            </div>

            <div class="column is-4-desktop is-6-tablet">
             <a href="https://chessaithon.pixel-online.org/">
              <div class="card">
                <div class="card-content">
                  <p class="title is-5">About</p>
                  <p class="">Visit the official web page of the project ChessAIthon
                  </p>
                </div>
              </div>
            </div>
           </a>
          </div>
        </div>
      </div>
    </section>
  </div>
`;
    }


}

customElements.define("chess-home", HomeComponent);