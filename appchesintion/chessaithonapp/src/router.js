export { routes, router }

const routes = new Map([
    ["#home", { component: "chess-home" }],
    ["#scenarios", { component: "chess-scenarios" }],
    ["#play", { component: "chess-play" }],
    ["#representation", { component: "chess-representation" }],
])

const router = (hash) => {
    console.log(hash);
    const [hashname,id] = hash.split('/'); 
    
    if(routes.has(hashname)){
        const {component} = routes.get(hashname);
        const webComponent = document.createElement(component);
        webComponent.identificator = id;
        document.querySelector('#content').replaceChildren(webComponent);
    }
}