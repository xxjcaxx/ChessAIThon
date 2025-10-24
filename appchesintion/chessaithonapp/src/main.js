import "./components/navbar/navbar.js";
import "./components/home/home.js";
import "./components/scenarios/scenarios.js";
import "./components/board/board.js";
import "./components/play/play.js";
import "./components/players/players.js";
import "./components/representation/representation.js";
import { initChessmarroBoard } from "chessmarro-board";
import { router } from "./router.js";

initChessmarroBoard();

console.log("main");

document.addEventListener("DOMContentLoaded", () => {
  //window.location.hash = "#home";
  router(window.location.hash);
  window.addEventListener("hashchange", () => {
    router(window.location.hash);
    });

});