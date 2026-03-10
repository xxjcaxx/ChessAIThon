import json

def json_to_forest(node, depth_limit=3, highlight=False):
    """
    Convierte un nodo del JSON de MCTS a formato de árbol 'forest' de LaTeX.
    """
    move = node.get("move", "None")
    # Escapar 'None' o movimientos vacíos para LaTeX
    if move == "None" or not move:
        move = "Raíz"
    
    visits = node.get("visits", 0)
    value = node.get("value", 0.0)
    
    # Formatear el contenido del nodo
    # Usamos llaves para permitir saltos de línea \\
    node_str = f"{{{move} \\\\ $N={visits}$ \\\\ $V={value:.2f}$}}"
    node_style = ", fill=green!25" if highlight else ""
    
    # Si llegamos al límite de profundidad, no procesamos hijos
    if depth_limit == 0:
        return f"[{node_str}{node_style}]"
    
    children_str = ""
    children = node.get("children", [])
    
    if children:
        # Ordenamos hijos por visitas para que el árbol sea más legible
        children = sorted(children, key=lambda x: x.get("visits", 0), reverse=True)
        max_visits = children[0].get("visits", 0)
        
        for child in children:
            is_best_child = child.get("visits", 0) == max_visits
            children_str += "\n" + "    " + json_to_forest(child, depth_limit - 1, highlight=is_best_child)
            
    return f"[{node_str}{node_style}{children_str}]"

def generate_latex_document(json_data, max_depth=3):
    """Genera el documento completo de LaTeX."""
    tree_data = json_data.get("mcts_tree", json_data)
    forest_tree = json_to_forest(tree_data, depth_limit=max_depth)
    
    template = r"""\documentclass[border=10pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{forest}

\forestset{
    mcts/.style={
        for tree={
            grow'=east,
            draw,
            rounded corners,
            node font=\scriptsize\sffamily,
            align=center,
            fill=white,
            parent anchor=east,
            child anchor=west,
            edge={draw, -latex},
            inner sep=1pt,
            l sep=4mm,
            s sep=2mm,
        }
    }
}

\begin{document}
    
\begin{forest}
    mcts
    """ + forest_tree + r"""
\end{forest}
    
\end{document}
"""
    return template

# --- USO DEL SCRIPT ---

# Cargar tu ejemplo
with open('output.example.json', 'r') as f:
    data = json.load(f)

# Generar para los 2 primeros niveles (Raíz + hijos directos)
latex_code = generate_latex_document(data, max_depth=2)

# Guardar o imprimir
with open('mcts_tree.tex', 'w') as f:
    f.write(latex_code)

print("Archivo 'mcts_tree.tex' generado con éxito.")