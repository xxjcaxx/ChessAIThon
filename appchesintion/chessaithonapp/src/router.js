export { routes, router }

const routes = new Map([
    ["#home", { component: "chess-home" }],
    ["#scenarios", { component: "chess-scenarios" }],
])

const router = (hash) => {

    if(routes.has(hash)){
        const {component} = routes.get(hash);
        const webComponent = document.createElement(component);
        document.querySelector('#content').replaceChildren(webComponent);
    }
}