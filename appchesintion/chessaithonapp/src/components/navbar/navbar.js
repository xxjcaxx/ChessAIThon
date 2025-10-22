class NavbarComponent extends HTMLElement {


    connectedCallback() {
        this.innerHTML = `
        <nav class="navbar is-dark is-fixed-top" role="navigation" aria-label="main navigation">
    <div class="navbar-brand">
      <a class="navbar-item" href="#home"><img src="/iconoblanco.svg"> Chess Minds</a>

        <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false" data-target="navbarChess">
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
          <span aria-hidden="true"></span>
        </a>
    </div> 

    <div id="navbarChess" class="navbar-menu">
      <div class="navbar-start">
      <a class="navbar-item" href="#home" class="is-active">Home</a>
      <a class="navbar-item" href="#scenarios">Chess scenarios</a>
      <a class="navbar-item"  href="#play">Play Chess</a>
      <a class="navbar-item"  href="#representation">Chess representation</a>
      <a class="navbar-item"  href="#ai">Chess AI</a>
      <a class="navbar-item" href="https://chessaithon.pixel-online.org/">About</a>
      </div>

    </div>


  </nav>
        `;


        const navbarBurguer = this.querySelector(".navbar-burger");
        const navbarMenu = this.querySelector(".navbar-menu");
        navbarBurguer.addEventListener("click", () => {
            navbarMenu.classList.toggle("is-active");
        });
    }

}

customElements.define("chess-navbar", NavbarComponent);