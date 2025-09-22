class SidebarComponent extends HTMLElement {

    connectedCallback() {
        this.innerHTML = `
        <aside class="menu sidebar" id="sidebar">
    <p class="menu-label">Menú</p>
    <ul class="menu-list">
      <li><a href="#home" class="is-active">Home</a></li>
      <li><a href="#scenarios">Chess scenarios</a></li>
      <li><a>Play Chess</a></li>
      <li><a>Chess representation</a></li>
      <li><a>Chess AI</a></li>
    </ul>
  </aside>
`;
    }


}

customElements.define("chess-sidebar", SidebarComponent);
