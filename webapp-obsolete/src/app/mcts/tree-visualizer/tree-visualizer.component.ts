import { AfterViewInit, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { example } from './tree-example';
import { CommonModule } from '@angular/common';
import * as d3 from 'd3';

interface TreeNode {
  state: number[][];
  actionTaken: number | null;
  value: number;
  visits: number;
  expandableMoves: number[];
  children?: TreeNode[];
}

function translatePlayer(player: number): string{
  if(player == 0) { return ' '};
  if(player == 1) { return '1'};
  if(player == -1) { return 'X'};
  return ' ';
}

@Component({
    selector: 'app-tree-visualizer',
    imports: [CommonModule],
    templateUrl: './tree-visualizer.component.html',
    styleUrl: './tree-visualizer.component.css'
})
export class TreeVisualizerComponent implements OnInit, AfterViewInit  {

  // https://speakerdeck.com/danielepolencic/visualising-game-trees-with-d3-dot-js?slide=88


  constructor(private elementRef: ElementRef<HTMLElement>){

  }

  @ViewChild('tree', { static: false }) divHello: ElementRef | undefined;

 ngAfterViewInit() {
   //this.divHello!.nativeElement.innerHTML = "Hello Angular";
   this.setupD3(this.divHello!.nativeElement);
 }


  example: TreeNode = example;




  setupD3(container: HTMLElement) {
    const width = 928;


    render_data(this.example);


    function render_data(data: TreeNode) {
      const root = d3.hierarchy(data);
      const dx = 60;
      const dy = width / (root.height + 1);

      const tree = d3.tree<TreeNode>().nodeSize([dx, dy]);

      root.sort((a, b) => d3.ascending(a.data.actionTaken || 0, b.data.actionTaken || 0));
      tree(root);

      let x0 = Infinity;
      let x1 = -x0;
      tree(root).each((d: d3.HierarchyPointNode<TreeNode>) => {
        if (d.x > x1) x1 = d.x;
        if (d.x < x0) x0 = d.x;
      });

      const height = x1 - x0 + dx * 2;

      const svg = d3.create("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-dy / 3, x0 - dx, width, height])
        .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;");

      const link = svg.append("g")
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-opacity", 0.4)
        .attr("stroke-width", 1.5)
        .selectAll()
        .data(root.links())
        .join("path")
        .attr("d", (d) => {
          const linkGenerator = d3
            .linkHorizontal<d3.HierarchyPointLink<TreeNode>, [number, number]>()
            .source((d) => [d.source.y, d.source.x])
            .target((d) => [d.target.y, d.target.x]);
          return linkGenerator(d as d3.HierarchyPointLink<TreeNode>);
        });

      const node = svg.append("g")
        .attr("stroke-linejoin", "round")
        .attr("stroke-width", 3)
        .selectAll()
        .data(tree(root).descendants())
        .join("g")
        .attr("transform", d => `translate(${d.y},${d.x})`);

    /*  node.append("circle")
        .attr("fill", d => d.children ? "#555" : "#999")
        .attr("r", 2.5);*/

        node.append("rect")
          .attr("width", 50) // ajusta el ancho según sea necesario
          .attr("height", 50) // ajusta la altura según sea necesario
          .attr("x", -25) // centrar el cuadrado horizontalmente
          .attr("y", -25) // centrar el cuadrado verticalmente
          .style("fill", "lightblue"); // color del cuadrado

    /*  node.append("text")
        .attr("dy", "0.31em")
        .attr("x", d => d.children ? -30 : 6)
        .attr("text-anchor", d => d.children ? "end" : "start")
        .text(d => `${d.data.value}`)
        .clone(true).lower()
        .attr("stroke", "white");*/


        node.each(function(d) {
          for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
              d3.select(this).append("text")
                .text(translatePlayer(d.data.state[i][j]))
                .attr("x", -18 + j * 18) // ajusta la posición x del texto
                .attr("y", -18 + i * 18) // ajusta la posición y del texto
                .attr("dy", "0.35em") // ajusta el desplazamiento vertical del texto
                .style("text-anchor", "middle") // ancla el texto en el medio
                .style("fill", "black"); // color del texto
            }
          }
        });

      container.append(svg.node()!);
    }
  }



  ngOnInit(): void {
    //const elementId = this.elementRef.nativeElement.id;
    //
  }



}
