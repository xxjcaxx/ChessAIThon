#!/bin/bash

# Ficheros de entrada y salida
INPUT="fens.txt"
OUTPUT="resultados.txt"

# Ruta a Stockfish
#STOCKFISH="./stockfish"  # Cambia según donde esté el binario

# Limpiamos fichero de salida
echo -e "FEN\tStockfish\tMCTS" > "$OUTPUT"

# Recorremos cada línea de FEN
while IFS= read -r fen; do
    # --- Stockfish ---
    # Ejecutamos Stockfish en depth 15 y MultiPV 1
    stockfish_move=$((echo -e "uci\nucinewgame\nisready\nposition fen $fen\ngo depth 15"; sleep 2; echo "quit") | stockfish | grep "bestmove" | awk '{print $2}'
)  # La columna 2 es la jugada en UCI

    # --- MCTS motor vía curl ---
    mcts_move=$(curl -s -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        -d "{\"fen\": \"$fen\", \"simulations\": 200}" \
        | jq -r '.move[0][1]')  # Usamos jq para extraer la jugada

    # --- Guardamos en fichero ---
    echo -e "$fen\t$stockfish_move\t$mcts_move" >> "$OUTPUT"

done < "$INPUT"

echo "Resultados guardados en $OUTPUT"
