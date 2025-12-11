import * as d3 from "d3";
import { exampleMCTSTree } from "./exemple";
import style from "./visualizer.css?inline"
import { initStyle, initTemplate } from '../componentsUtils.js';


class mctsVisualizer extends HTMLElement {


    constructor() {
        super();
        this.shadow = this.attachShadow({ mode: 'open' });
    }



    connectedCallback() {
        this.shadow.append(
            initStyle(style),
            initTemplate(`
                <div>
                <button id="toggleButton" class="button is-primary">
                    Show / Hide Data
                </button>
                <pre id="treeDataDisplay" class="is-hidden"></pre>
                <div id="treeVisualizer"></div>
                </div>
                `)
        );

        this.treeDataDisplay = this.shadow.querySelector("#treeDataDisplay");
        this.treeVisualizer = this.shadow.querySelector("#treeVisualizer");


        const toggleButton = this.shadow.getElementById('toggleButton');
        toggleButton.addEventListener('click', () => {
            this.treeDataDisplay.classList.toggle('is-hidden');
        });
        this.treeData = exampleMCTSTree;
    }

    set treeData(jsonStructure) {
        this._treeData = jsonStructure;
        this.treeDataDisplay.textContent = JSON.stringify(jsonStructure, null, 2);
        const treeSvg = this.renderTree(jsonStructure);
        this.treeVisualizer.replaceChildren(treeSvg);
    }

    get treeData() {
        return this._treeData;
    }


    renderTree(treeData) {
        // Specify the charts’ dimensions. The height is variable, depending on the layout.
        const width = 928;
        const marginTop = 30;
        const marginRight = 10;
        const marginBottom = 30;
        const marginLeft = 60;

        // --- Constantes para la Caja del Nodo ---
        const nodeWidth = 100;
        const nodeHeight = 50;
        const cornerRadius = 6;
        const lineHeight = 12;

        
        const tooltipSize = 150; // Tamaño (ancho y alto) del cuadrado de la tooltip
        const tooltipMargin = 5; // Margen para posicionarlo dentro de la caja principal
        // ---------------------------------------------

        // Ajuste el espaciado vertical (dx) y horizontal (dy)
        const root = d3.hierarchy(treeData);
        const dx = nodeHeight + 5; // Nueva separación vertical
        const dy = (width - marginRight - marginLeft) / (1 + root.height);

        // Define el layout del árbol y la forma de los enlaces (paths)
        const tree = d3.tree().nodeSize([dx, dy]);
        const diagonal = d3.linkHorizontal().x(d => d.y).y(d => d.x);

        // Crea el contenedor SVG
        const svg = d3.create("svg")
            .attr("width", width)
            .attr("height", dx)
            .attr("viewBox", [-marginLeft, -marginTop, width, dx])
            .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif; user-select: none;");

        const defs = svg.append("defs");
        
        // Creamos el patrón y la imagen. Usaremos el ID 'chess-tooltip-pattern' 
        // y su atributo 'xlink:href' será actualizado dinámicamente.
        defs.append("pattern")
            .attr("id", "chess-tooltip-pattern") 
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 1) 
            .attr("height", 1)
            .append("image")
                .attr("id", "tooltip-pattern-image") // ID para la imagen dentro del patrón
                .attr("width", tooltipSize)    
                .attr("height", tooltipSize)
                .attr("preserveAspectRatio", "xMidYMid slice");

        const patternImage = defs.select("#tooltip-pattern-image");

        const gLink = svg.append("g")
            .attr("fill", "none")
            .attr("stroke", "#555")
            .attr("stroke-opacity", 0.4)
            .attr("stroke-width", 1.5);

        const gNode = svg.append("g")
            .attr("cursor", "pointer")
            .attr("pointer-events", "all");

        const getChildrenCount = d => {
            return (d.children ? d.children.length : 0) + (d._children ? d._children.length : 0);
        };


        function update(event, source) {
            const duration = event?.altKey ? 2500 : 250;
            const nodes = root.descendants().reverse();
            const links = root.links();

            // Computa el nuevo layout del árbol.
            tree(root);

            let left = root;
            let right = root;
            root.eachBefore(node => {
                if (node.x < left.x) left = node;
                if (node.x > right.x) right = node;
            });

            const height = right.x - left.x + marginTop + marginBottom;

            const transition = svg.transition()
                .duration(duration)
                .attr("height", height)
                .attr("viewBox", [-marginLeft, left.x - marginTop, width, height])
                .tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));

            // Update the nodes…
            const node = gNode.selectAll("g")
                .data(nodes, d => d.id);

            // Enter any new nodes at the parent's previous position (source.y0, source.x0).
            const nodeEnter = node.enter().append("g")
                // Aquí es donde source.y0 y source.x0 deben estar definidas
                .attr("transform", d => `translate(${source.y0},${source.x0})`)
                .attr("fill-opacity", 0)
                .attr("stroke-opacity", 0)
                .on("click", (event, d) => {
                    d.children = d.children ? null : d._children;
                    update(event, d);
                    // Aquí es donde iría this.updateChessboard(d.data.fen_pos); 
                });

            // 1. Dibuja el rectángulo de fondo con bordes redondeados
            nodeEnter.append("rect")
                .attr("x", -nodeWidth / 2)
                .attr("y", -nodeHeight / 2)
                .attr("width", nodeWidth)
                .attr("height", nodeHeight)
                .attr("rx", cornerRadius)
                .attr("ry", cornerRadius)
                .attr("fill", d => d._children ? "#607D8B" : "#CFD8DC")
                .attr("stroke", "#333")
                .attr("stroke-width", 1.5);

            // 2. Texto: Movimiento
            nodeEnter.append("text")
                .attr("x", 0)
                .attr("y", -nodeHeight / 2 + lineHeight)
                .attr("text-anchor", "middle")
                .attr("font-size", "9px")
                .attr("font-weight", "bold")
                .attr("fill", d => d._children ? "#FFF" : "#000")
                .text(d => d.data.move !== 'None' ? d.data.move : 'ROOT');



            // 3. Texto: Visits (N) y Score/Value (Q)
            nodeEnter.append("text")
                .attr("x", 0)
                .attr("y", -nodeHeight / 2 + lineHeight * 2.2)
                .attr("text-anchor", "middle")
                .attr("font-size", "9px")
                .attr("fill", d => d._children ? "#DDD" : "#333")
                .text(d => `Visits: ${d.data.visits} | Value: ${d.data.value ? d.data.value.toFixed(2) : 'N/A'}`);

            // 4. Texto: Children Count (C)
            nodeEnter.append("text")
                .attr("x", 0)
                .attr("y", -nodeHeight / 2 + lineHeight * 3.4)
                .attr("text-anchor", "middle")
                .attr("font-size", "9px")
                .attr("fill", d => d._children ? "#DDD" : "#333")
                .text(d => `Children: ${getChildrenCount(d)}`);


            nodeEnter.append("rect")
                .attr("class", "tooltip-square") // Clase para fácil selección
                .attr("x", nodeWidth / 2 
                   // - tooltipSize - tooltipMargin
                )
                .attr("y", -nodeHeight / 2 
                    //+ tooltipMargin
                )
                .attr("width", tooltipSize)
                .attr("height", tooltipSize)
                .attr("fill", "url(#chess-tooltip-pattern)") // Color de fondo llamativo (Amarillo)
                .attr("stroke", "#333")
                .attr("stroke-width", 1)
                .attr("opacity", 0); // ¡IMPORTANTE! Oculto inicialmente

            nodeEnter
                .on("mouseover", function (event, d) {
                    // Seleccionar el cuadrado dentro del nodo actual (<g>)

                    const imageURL = d.data.miniBoard;
                    patternImage.attr("xlink:href", imageURL);

                    d3.select(this).select(".tooltip-square")
                        .transition()
                        .duration(100)
                        .attr("opacity", 1); // Mostrar

                    // Opcional: Resaltar la caja principal
                    d3.select(this).select("rect:not(.tooltip-square)")
                        .attr("stroke", "#FF5722")
                        .attr("stroke-width", 2.5);
                })
                .on("mouseout", function (event, d) {
                    // Ocultar el cuadrado
                    d3.select(this).select(".tooltip-square")
                        .transition()
                        .duration(100)
                        .attr("opacity", 0);

                    // Opcional: Restaurar el borde de la caja principal
                    d3.select(this).select("rect:not(.tooltip-square)")
                        .attr("stroke", "#333")
                        .attr("stroke-width", 1.5);
                });

            // Transición de nodos a su nueva posición
            const nodeUpdate = node.merge(nodeEnter).transition(transition)
                .attr("transform", d => `translate(${d.y},${d.x})`)
                .attr("fill-opacity", 1)
                .attr("stroke-opacity", 1);

            nodeUpdate.select("rect")
                .attr("fill", d => d._children ? "#607D8B" : "#CFD8DC");

            // Transition exiting nodes to the parent's new position.
            const nodeExit = node.exit().transition(transition).remove()
                .attr("transform", d => `translate(${source.y},${source.x})`)
                .attr("fill-opacity", 0)
                .attr("stroke-opacity", 0);

            // Update the links…
            const link = gLink.selectAll("path")
                .data(links, d => d.target.id);

            // Enter any new links at the parent's previous position (source.x0, source.y0).
            const linkEnter = link.enter().append("path")
                .attr("d", d => {
                    const o = { x: source.x0, y: source.y0 };
                    return diagonal({ source: o, target: o });
                });

            // Transition links to their new position.
            link.merge(linkEnter).transition(transition)
                .attr("d", diagonal);

            // Transition exiting nodes to the parent's new position.
            link.exit().transition(transition).remove()
                .attr("d", d => {
                    const o = { x: source.x, y: source.y };
                    return diagonal({ source: o, target: o });
                });

            // Stash the old positions for transition.
            root.eachBefore(d => {
                d.x0 = d.x;
                d.y0 = d.y;
            });
        }

        // --- INICIALIZACIÓN CORREGIDA ---

        // 1. Aplicar la lógica de colapsado deseada: mostrar solo raíz y primer nivel (depth 0 y 1).
        root.descendants().forEach((d, i) => {
            d.id = i;
            d._children = d.children;
            // FIX #2: Colapsar todos los nodos con profundidad > 1
            if (d.depth > 0) d.children = null;
           
            
        });

        // 2. Calcular la posición inicial del árbol (basada en el estado colapsado).
        tree(root);

        // 3. FIX #1: Inicializar las posiciones 'previas' (x0, y0) con las posiciones 'actuales' (x, y).
        // Esto asegura que la primera transición de entrada (enter) de nodos y links sea correcta.
        root.eachBefore(d => {
            d.x0 = d.x;
            d.y0 = d.y;
        });

        // 4. Iniciar el árbol.
        // Pasamos 'root' como la fuente para que la transición 'enter' de los hijos funcione.
        update(null, root);

        return svg.node();
    }

}


customElements.define("chess-mcts-visualizer", mctsVisualizer);