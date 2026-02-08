var m = Object.defineProperty;
var u = (r, e, o) => e in r ? m(r, e, { enumerable: !0, configurable: !0, writable: !0, value: o }) : r[e] = o;
var h = (r, e, o) => u(r, typeof e != "symbol" ? e + "" : e, o);
const G = (r, e) => r + e, X = () => Array(8).fill(0).map(() => Array(8).fill(0)), f = (r) => r.split("").map(
  (e) => isNaN(parseInt(e)) ? e : Array(parseInt(e)).fill(0)
).flat().filter((e) => e != "/").reduce(
  (e, o, s) => (s < 64 && e[Math.floor(s / 8)].push(o), e),
  [[], [], [], [], [], [], [], []]
), Y = () => f("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), U = (r) => {
  let e = [];
  for (let o of r) {
    let s = 0;
    for (let i of o)
      i != 0 && s > 0 && (e.push(s), s = 0), i != 0 && e.push(i), i === 0 && s++;
    s > 0 && e.push(s), e.push("/");
  }
  return e.pop(), e.join("");
}, b = (r) => ([e, o], [s, i]) => {
  const a = structuredClone(r), n = a[o][e];
  return a[o][e] = 0, a[i][s] = n, a;
}, _ = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill='none'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20fill='%23fff'%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M9%2036c3.39-.97%2010.11.43%2013.5-2%203.39%202.43%2010.11%201.03%2013.5%202%200%200%201.65.54%203%202-.68.97-1.65.99-3%20.5-3.39-.97-10.11.46-13.5-1-3.39%201.46-10.11.03-13.5%201-1.35.49-2.32.47-3-.5%201.35-1.46%203-2%203-2zm6-4c2.5%202.5%2012.5%202.5%2015%200%20.5-1.5%200-2%200-2%200-2.5-2.5-4-2.5-4%205.5-1.5%206-11.5-5-15.5-11%204-10.5%2014-5%2015.5%200%200-2.5%201.5-2.5%204%200%200-.5.5%200%202z'/%3e%3cpath%20d='M25%208a2.5%202.5%200%201%201-5%200%202.5%202.5%200%201%201%205%200z'/%3e%3c/g%3e%3cpath%20d='M17.5%2026h10M15%2030h15m-7.5-14.5v5M20%2018h5'%20stroke-linecap='round'%20stroke-width='1.5'/%3e%3c/svg%3e", k = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: _
}, Symbol.toStringTag, { value: "Module" })), w = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill='none'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20stroke-linecap='round'%20stroke-width='1.5'%3e%3cpath%20d='M22.5%2011.63V6'/%3e%3cpath%20d='M20%208h5'/%3e%3c/g%3e%3cpath%20d='M22.5%2025s4.5-7.5%203-10.5c0%200-1-2.5-3-2.5s-3%202.5-3%202.5c-1.5%203%203%2010.5%203%2010.5'%20stroke-width='1.5'%20fill='%23fff'/%3e%3cg%20stroke-width='1.5'%20stroke-linecap='round'%20stroke-linejoin='round'%3e%3cpath%20d='M12.5%2037c5.5%203.5%2014.5%203.5%2020%200v-7s9-4.5%206-10.5c-4-6.5-13.5-3.5-16%204V27v-3.5c-2.5-7.5-12-10.5-16-4-3%206%206%2010.5%206%2010.5v7'%20fill='%23fff'/%3e%3cpath%20d='M12.5%2030c5.5-3%2014.5-3%2020%200m-20%203.5c5.5-3%2014.5-3%2020%200m-20%203.5c5.5-3%2014.5-3%2020%200'/%3e%3c/g%3e%3c/svg%3e", M = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: w
}, Symbol.toStringTag, { value: "Module" })), y = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20fill='%23fff'%20stroke-linecap='round'%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M22%2010c10.5%201%2016.5%208%2016%2029H15c0-9%2010-6.5%208-21'/%3e%3cpath%20d='M24%2018c.38%202.91-5.55%207.37-8%209-3%202-2.82%204.34-5%204-1.042-.94%201.41-3.04%200-3-1%200%20.19%201.23-1%202-1%200-4.003%201-4-4%200-2%206-12%206-12s1.89-1.9%202-3.5c-.73-.994-.5-2-.5-3%201-1%203%202.5%203%202.5h2s.78-1.992%202.5-3c1%200%201%203%201%203'/%3e%3c/g%3e%3cpath%20d='M9.5%2025.5a.5.5%200%201%201-1%200%20.5.5%200%201%201%201%200zm5.433-9.75c-.414.717-.944%201.187-1.183%201.049s-.097-.832.317-1.549.944-1.187%201.183-1.049.097.832-.317%201.549z'%20stroke-linejoin='round'%20stroke-width='1.5'/%3e%3c/svg%3e", x = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: y
}, Symbol.toStringTag, { value: "Module" })), j = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20xmlns:v='https://vecta.io/nano'%3e%3cpath%20d='M22.5%209a4%204%200%200%200-4%204c0%20.89.29%201.71.78%202.38C17.33%2016.5%2016%2018.59%2016%2021c0%202.03.94%203.84%202.41%205.03-3%201.06-7.41%205.55-7.41%2013.47h23c0-7.92-4.41-12.41-7.41-13.47%201.47-1.19%202.41-3%202.41-5.03%200-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38a4%204%200%200%200-4-4z'%20fill='%23fff'%20stroke='%23000'%20stroke-width='1.5'/%3e%3c/svg%3e", S = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: j
}, Symbol.toStringTag, { value: "Module" })), $ = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill='%23fff'%20stroke='%23000'%20stroke-linejoin='round'%20stroke-width='1.5'%20xmlns:v='https://vecta.io/nano'%3e%3cpath%20d='M9%2026c8.5-1.5%2021-1.5%2027%200l2.5-12.5L31%2025l-.3-14.1-5.2%2013.6-3-14.5-3%2014.5-5.2-13.6L14%2025%206.5%2013.5%209%2026z'/%3e%3cpath%20d='M9%2026c0%202%201.5%202%202.5%204%201%201.5%201%201%20.5%203.5-1.5%201-1%202.5-1%202.5-1.5%201.5%200%202.5%200%202.5%206.5%201%2016.5%201%2023%200%200%200%201.5-1%200-2.5%200%200%20.5-1.5-1-2.5-.5-2.5-.5-2%20.5-3.5%201-2%202.5-2%202.5-4-8.5-1.5-18.5-1.5-27%200z'/%3e%3cpath%20d='M11.5%2030c3.5-1%2018.5-1%2022%200M12%2033.5c6-1%2015-1%2021%200'%20fill='none'/%3e%3ccircle%20cx='6'%20cy='12'%20r='2'/%3e%3ccircle%20cx='14'%20cy='9'%20r='2'/%3e%3ccircle%20cx='22.5'%20cy='8'%20r='2'/%3e%3ccircle%20cx='31'%20cy='9'%20r='2'/%3e%3ccircle%20cx='39'%20cy='12'%20r='2'/%3e%3c/svg%3e", z = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: $
}, Symbol.toStringTag, { value: "Module" })), O = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill='%23fff'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M9%2039h27v-3H9v3z'/%3e%3cpath%20d='M12%2036v-4h21v4H12z'/%3e%3cpath%20d='M11%2014V9h4v2h5V9h5v2h5V9h4v5'%20stroke-linecap='butt'/%3e%3cpath%20d='M34%2014l-3%203H14l-3-3'%20stroke-linecap='round'/%3e%3c/g%3e%3cpath%20d='M31%2017v12.5H14V17'%20stroke-linecap='butt'%20stroke-linejoin='miter'%20stroke-width='1.5'/%3e%3cg%20stroke-linecap='round'%3e%3cpath%20d='M31%2029.5l1.5%202.5h-20l1.5-2.5'%20stroke-linejoin='round'%20stroke-width='1.5'/%3e%3cpath%20d='M11%2014h23'%20fill='none'%20stroke-linejoin='miter'%20stroke-width='1.5'/%3e%3c/g%3e%3c/svg%3e", C = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: O
}, Symbol.toStringTag, { value: "Module" })), P = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill='none'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20fill='%23000'%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M9%2036c3.39-.97%2010.11.43%2013.5-2%203.39%202.43%2010.11%201.03%2013.5%202%200%200%201.65.54%203%202-.68.97-1.65.99-3%20.5-3.39-.97-10.11.46-13.5-1-3.39%201.46-10.11.03-13.5%201-1.35.49-2.32.47-3-.5%201.35-1.46%203-2%203-2zm6-4c2.5%202.5%2012.5%202.5%2015%200%20.5-1.5%200-2%200-2%200-2.5-2.5-4-2.5-4%205.5-1.5%206-11.5-5-15.5-11%204-10.5%2014-5%2015.5%200%200-2.5%201.5-2.5%204%200%200-.5.5%200%202z'/%3e%3cpath%20d='M25%208a2.5%202.5%200%201%201-5%200%202.5%202.5%200%201%201%205%200z'/%3e%3c/g%3e%3cpath%20d='M17.5%2026h10M15%2030h15m-7.5-14.5v5M20%2018h5'%20stroke='%23fff'%20stroke-linecap='round'%20stroke-width='1.5'/%3e%3c/svg%3e", T = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: P
}, Symbol.toStringTag, { value: "Module" })), E = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill='none'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20stroke-linejoin='miter'%20stroke-width='1.5'%3e%3cpath%20d='M22.5%2011.63V6'%20stroke-linecap='round'/%3e%3cpath%20d='M22.5%2025s4.5-7.5%203-10.5c0%200-1-2.5-3-2.5s-3%202.5-3%202.5c-1.5%203%203%2010.5%203%2010.5'%20stroke-linecap='butt'%20fill='%23000'/%3e%3c/g%3e%3cg%20stroke-linecap='round'%3e%3cpath%20d='M12.5%2037c5.5%203.5%2014.5%203.5%2020%200v-7s9-4.5%206-10.5c-4-6.5-13.5-3.5-16%204V27v-3.5c-2.5-7.5-12-10.5-16-4-3%206%206%2010.5%206%2010.5v7'%20fill='%23000'%20stroke-linejoin='round'%20stroke-width='1.5'/%3e%3cpath%20d='M20%208h5'%20stroke-linejoin='miter'%20stroke-width='1.5'/%3e%3cpath%20d='M32%2029.5s8.5-4%206.03-9.65C34.15%2014%2025%2018%2022.5%2024.5v2.1-2.1C20%2018%2010.85%2014%206.97%2019.85%204.5%2025.5%2013%2029.5%2013%2029.5m-.5.5c5.5-3%2014.5-3%2020%200m-20%203.5c5.5-3%2014.5-3%2020%200m-20%203.5c5.5-3%2014.5-3%2020%200'%20stroke-width='1.5'%20stroke-linejoin='round'%20stroke='%23fff'/%3e%3c/g%3e%3c/svg%3e", H = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: E
}, Symbol.toStringTag, { value: "Module" })), B = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20fill='%23000'%20stroke-linecap='round'%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M22%2010c10.5%201%2016.5%208%2016%2029H15c0-9%2010-6.5%208-21'/%3e%3cpath%20d='M24%2018c.38%202.91-5.55%207.37-8%209-3%202-2.82%204.34-5%204-1.042-.94%201.41-3.04%200-3-1%200%20.19%201.23-1%202-1%200-4.003%201-4-4%200-2%206-12%206-12s1.89-1.9%202-3.5c-.73-.994-.5-2-.5-3%201-1%203%202.5%203%202.5h2s.78-1.992%202.5-3c1%200%201%203%201%203'/%3e%3c/g%3e%3cg%20fill='%23fff'%3e%3cpath%20d='M9.5%2025.5a.5.5%200%201%201-1%200%20.5.5%200%201%201%201%200zm5.433-9.75c-.414.717-.944%201.187-1.183%201.049s-.097-.832.317-1.549.944-1.187%201.183-1.049.097.832-.317%201.549z'%20stroke-linejoin='round'%20stroke-width='1.5'%20stroke='%23fff'/%3e%3cpath%20d='M24.55%2010.4l-.45%201.45.5.15c3.15%201%205.65%202.49%207.9%206.75S35.75%2029.06%2035.25%2039l-.05.5h2.25l.05-.5c.5-10.06-.88-16.85-3.25-21.34s-5.79-6.64-9.19-7.16l-.51-.1z'%20stroke='none'/%3e%3c/g%3e%3c/svg%3e", A = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: B
}, Symbol.toStringTag, { value: "Module" })), L = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20xmlns:v='https://vecta.io/nano'%3e%3cpath%20d='M22.5%209a4%204%200%200%200-4%204c0%20.89.29%201.71.78%202.38C17.33%2016.5%2016%2018.59%2016%2021c0%202.03.94%203.84%202.41%205.03-3%201.06-7.41%205.55-7.41%2013.47h23c0-7.92-4.41-12.41-7.41-13.47%201.47-1.19%202.41-3%202.41-5.03%200-2.41-1.33-4.5-3.28-5.62.49-.67.78-1.49.78-2.38a4%204%200%200%200-4-4z'%20stroke='%23000'%20stroke-width='1.5'/%3e%3c/svg%3e", V = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: L
}, Symbol.toStringTag, { value: "Module" })), N = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M9%2026c8.5-1.5%2021-1.5%2027%200l2.5-12.5L31%2025l-.3-14.1-5.2%2013.6-3-14.5-3%2014.5-5.2-13.6L14%2025%206.5%2013.5%209%2026z'/%3e%3cpath%20d='M9%2026c0%202%201.5%202%202.5%204%201%201.5%201%201%20.5%203.5-1.5%201-1%202.5-1%202.5-1.5%201.5%200%202.5%200%202.5%206.5%201%2016.5%201%2023%200%200%200%201.5-1%200-2.5%200%200%20.5-1.5-1-2.5-.5-2.5-.5-2%20.5-3.5%201-2%202.5-2%202.5-4-8.5-1.5-18.5-1.5-27%200z'/%3e%3cpath%20d='M11.5%2030c3.5-1%2018.5-1%2022%200M12%2033.5c6-1%2015-1%2021%200'%20stroke-linecap='round'/%3e%3ccircle%20cx='6'%20cy='12'%20r='2'/%3e%3ccircle%20cx='14'%20cy='9'%20r='2'/%3e%3ccircle%20cx='22.5'%20cy='8'%20r='2'/%3e%3ccircle%20cx='31'%20cy='9'%20r='2'/%3e%3ccircle%20cx='39'%20cy='12'%20r='2'/%3e%3cpath%20d='M11%2038.5a35%2035%201%200%200%2023%200'%20stroke-linecap='butt'%20fill='none'/%3e%3c/g%3e%3cpath%20d='M11%2029a35%2035%201%200%201%2023%200m-21.5%202.5h20m-21%203a35%2035%201%200%200%2022%200m-23%203a35%2035%201%200%200%2024%200'%20fill='none'%20stroke='%23fff'%20stroke-linecap='round'%20stroke-linejoin='round'%20stroke-width='1.5'/%3e%3c/svg%3e", q = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: N
}, Symbol.toStringTag, { value: "Module" })), I = "data:image/svg+xml,%3csvg%20xmlns='http://www.w3.org/2000/svg'%20width='45'%20height='45'%20fill-rule='evenodd'%20stroke='%23000'%20xmlns:v='https://vecta.io/nano'%3e%3cg%20stroke-linejoin='round'%20stroke-width='1.5'%3e%3cpath%20d='M9%2039h27v-3H9v3zm3.5-7l1.5-2.5h17l1.5%202.5h-20z'/%3e%3cpath%20d='M12%2036v-4h21v4H12z'/%3e%3c/g%3e%3cpath%20d='M14%2029.5v-13h17v13H14z'%20stroke-linejoin='miter'%20stroke-width='1.5'/%3e%3cg%20stroke-linejoin='round'%3e%3cpath%20d='M14%2016.5L11%2014h23l-3%202.5H14z'%20stroke-width='1.5'/%3e%3cpath%20d='M11%2014V9h4v2h5V9h5v2h5V9h4v5H11z'%20stroke-width='1.5'/%3e%3c/g%3e%3cpath%20d='M12%2035.5h21m-20-4h19m-18-2h17m-17-13h17M11%2014h23'%20stroke-linejoin='miter'%20fill='none'%20stroke='%23fff'%20stroke-width='1'%20stroke-linecap='round'/%3e%3c/svg%3e", D = /* @__PURE__ */ Object.freeze(/* @__PURE__ */ Object.defineProperty({
  __proto__: null,
  default: I
}, Symbol.toStringTag, { value: "Module" })), R = "#app{color:#2c3e50;font-family:Avenir,Helvetica,Arial,sans-serif;-moz-osx-font-smoothing:grayscale;-webkit-font-smoothing:antialiased;margin-top:60px;text-align:center}#divBoard{aspect-ratio:1 / 1;display:grid;width:100%}.piece{align-items:center;display:flex;font-family:Gill Sans,Gill Sans MT,Calibri,Trebuchet MS,sans-serif;font-size:3vw;font-weight:bolder;grid-column:span 3;grid-row:span 3;justify-content:center}.piece img{transition:transform .5s ease-in-out;width:100%}.piece.numbers{color:#fff;grid-column:span 1}.piece.letters{color:#fff;grid-row:span 1}", F = /* @__PURE__ */ Object.assign({
  "./pieces/B.svg": k,
  "./pieces/K.svg": M,
  "./pieces/N.svg": x,
  "./pieces/P.svg": S,
  "./pieces/Q.svg": z,
  "./pieces/R.svg": C,
  "./pieces/b.svg": T,
  "./pieces/k.svg": H,
  "./pieces/n.svg": A,
  "./pieces/p.svg": V,
  "./pieces/q.svg": q,
  "./pieces/r.svg": D
}), d = [" ", "a", "b", "c", "d", "e", "f", "g", "h", " "], J = (r) => {
  const [e, o, s, i] = r.split("");
  return [
    d.indexOf(e) - 1,
    8 - parseInt(o),
    d.indexOf(s) - 1,
    8 - parseInt(i)
  ];
};
class K extends HTMLElement {
  constructor() {
    super();
    h(this, "board");
    h(this, "chessImages", Object.fromEntries(
      Object.entries(F).map(([o, s]) => [
        `${o}`.split("/")[2].split(".")[0],
        s.default
      ])
    ));
    h(this, "blackColor", "#b58863");
    h(this, "whiteColor", "#f0d9b5");
    h(this, "numbers", !1);
    h(this, "numbersColor", "black");
    h(this, "legalMoves", []);
    this.board = Array(8).fill(0).map(() => Array(8).fill(0));
  }
  connectedCallback() {
    this.dataset.fen && (this.board = f(this.dataset.fen)), this.dataset.blackColor && (this.blackColor = this.dataset.blackColor), this.dataset.whiteColor && (this.whiteColor = this.dataset.whiteColor), this.dataset.numbers && (this.numbers = this.dataset.numbers === "true"), this.dataset.numbersColor && (this.numbersColor = this.dataset.numbersColor), this.dataset.legalMoves && (this.legalMoves = JSON.parse(this.dataset.legalMoves).map(J)), this.render();
  }
  disconnectedCallback() {
  }
  adoptedCallback() {
  }
  render() {
    const o = this.shadowRoot ? this.shadowRoot : this.attachShadow({ mode: "open" }), s = document.createDocumentFragment(), i = document.createElement("div");
    i.id = "divBoard", i.style.opacity = "0";
    const a = new CSSStyleSheet();
    a.replace(R).then(() => {
      a.insertRule(`.black{background-color: ${this.blackColor};}`), a.insertRule(`.white{background-color: ${this.whiteColor};}`), a.insertRule(`#divBoard {
                                        grid-template-columns: repeat(${this.numbers ? 26 : 24}, 1fr);
                                        grid-template-rows: repeat(${this.numbers ? 26 : 24}, 1fr);
                                      }`), a.insertRule(
        `.piece.numbers, .piece.letters { background-color: ${this.numbersColor};}`
      );
    }), o.adoptedStyleSheets = [a], i.innerHTML = this.board ? (
      // Top Letters
      `${this.numbers ? d.map(
        (n) => `<div class="piece letters ${n != " " ? "" : "numbers"}">${n}</div>`
      ).join("") : ""}` + this.board.map(
        (n, t) => (
          //left numbers
          `${this.numbers ? `
          <div class="piece numbers">${8 - t}</div>
          ` : ""}` + // Pieces
          n.map(
            (l, c) => `
            <div class="piece
            ${(t + c) % 2 === 1 ? "black" : ""}
            ${(t + c) % 2 === 0 ? "white" : ""}
            "
            data-position="${d[c + 1]}${8 - t}"
            data-letter="${d[c + 1]}"
            data-number="${7 - t}"
            data-matrix_position="[${c},${t}]"
            >${l != 0 ? `<img data-piece="${l}" data-position="${d[c + 1]}${8 - t}"
            data-letter="${d[c + 1]}"
            data-number="${8 - t}"
            data-matrix_position="[${c},${t}]"  src="${this.chessImages[l]}">` : ""}</div>
            `
          ).join("") + // right numbers
          `${this.numbers ? `
          <div class="piece numbers">${8 - t}</div>
          ` : ""}`
        )
      ).join("") + // Bottom Letters
      `${this.numbers ? d.map((n) => `<div class="piece letters ${n != " " ? "" : "numbers"}">${n}</div>`).join("") : ""}`
    ) : "", i.querySelectorAll("div.piece").forEach((n) => {
      const t = n;
      t.addEventListener("drop", (l) => {
        var g;
        l.preventDefault();
        const c = ((g = l.dataTransfer) == null ? void 0 : g.getData("application/json")) || "", p = JSON.parse(c);
        this.dispatchEvent(
          new CustomEvent("chessmarro-move", {
            bubbles: !0,
            composed: !0,
            detail: {
              origin: { ...p },
              destiny: {
                position: t.dataset.position,
                matrix_position: t.dataset.matrix_position
              },
              uci: p.position + t.dataset.position
            }
          })
        );
      }), t.addEventListener("dragover", (l) => {
        l.preventDefault();
      });
    }), i.querySelectorAll("img").forEach((n) => {
      n.addEventListener("dragstart", (t) => {
        t.dataTransfer && t.target instanceof HTMLImageElement && (t.dataTransfer.setData("text/plain", t.target.dataset.piece), t.dataTransfer.setData(
          "application/json",
          JSON.stringify({
            piece: t.target.dataset.piece,
            position: t.target.dataset.position,
            matrix_position: t.target.dataset.matrix_position
          })
        ), console.log(n));
      });
    }), s.appendChild(i), o.replaceChildren(s), requestAnimationFrame(() => {
      i.style.opacity = "1";
    });
  }
  movePiece([o, s], [i, a], n = 0.5) {
    const t = this.shadowRoot.querySelector(
      `img[data-position="${d[o + 1]}${8 - s}"]`
    ), l = this.shadowRoot.querySelector(
      `div[data-position="${d[i + 1]}${8 - a}"]`
    );
    if (t && l) {
      const c = t.getBoundingClientRect(), p = l.getBoundingClientRect(), g = p.left - c.left, v = p.top - c.top;
      t.style.transition = `transform ${n}s ease-in-out`, t.style.transform = `translate(${g}px, ${v}px)`, t.addEventListener(
        "transitionend",
        () => {
          this.board = b(this.board)([o, s], [i, a]), this.render();
        },
        { once: !0 }
      );
    }
  }
}
const W = (r = "chessmarro-board") => {
  customElements.define(r, K);
};
export {
  X as generateArrayBoard,
  U as getFen,
  W as initChessmarroBoard,
  b as movePiece,
  f as setFen,
  Y as setStartPosition,
  G as sum
};
