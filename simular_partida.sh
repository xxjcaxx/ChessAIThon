#!/bin/bash

# --- CONFIGURACIÓN ---
OUTPUT="partida_completa4.txt"
# FEN inicial (puedes cambiarlo por 'startpos' o cualquier FEN)
FEN_INICIAL=${1:-"r1bqk2r/pppp1ppp/2n2n2/2b1p3/2BPP3/2P2N2/PP3PPP/RNBQK2R b KQkq - 0 5"}
current_fen="$FEN_INICIAL"
MAX_MOVES=100  # Límite para evitar partidas infinitas

echo -e "Turno\tFEN\tStockfish_Move\tMCTS_Opinion\tMCTS_4000" > "$OUTPUT"

echo "Iniciando simulación de partida..."

for (( i=1; i<=$MAX_MOVES; i++ ))
do
    echo "Procesando turno $i..."

    # 1. Obtener la mejor jugada de Stockfish
    # Usamos el truco del sleep para asegurar profundidad 15
    sf_raw=$( (echo -e "uci\nucinewgame\nisready\nposition fen $current_fen\ngo depth 15"; sleep 2; echo "quit") | stockfish )
    
    sf_move=$(echo "$sf_raw" | grep "bestmove" | awk '{print $2}')
    echo "Stockfish output for turn %d:\n%s\n" "$i" "$sf_raw"
    # Si Stockfish no devuelve jugada (mate o tablas), terminamos
    if [[ "$sf_move" == "(none)" || -z "$sf_move" ]]; then
        echo "Fin de la partida (Mate o Tablas) en el turno $i."
        break
    fi

    # 2. Obtener la opinión de tu API (MCTS) para ese mismo FEN
    mcts_move=$(curl -s -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"fen\": \"$current_fen\", \"simulations\": 200}" \
    | jq -r '[.move, (.alternatives[] | .[0])] | join(",")')
    echo "MCTS output for turn %d: %s\n" "$i" "$mcts_move"
    mcts_move4000=$(curl -s -X POST "http://127.0.0.1:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"fen\": \"$current_fen\", \"simulations\": 400}" \
    | jq -r '[.move, (.alternatives[] | .[0])] | join(",")')
    echo "MCTS output for turn %d: %s\n" "$i" "$mcts_move4000"
    # 3. Guardar el estado actual antes de mover
    echo -e "$i\t$current_fen\t$sf_move\t$mcts_move\t$mcts_move4000" >> "$OUTPUT"

    # 4. ACTUALIZAR EL FEN:
    # Le pedimos a Stockfish que haga el movimiento y nos diga el nuevo FEN
    # El comando 'd' (display) en Stockfish imprime el FEN actual
    current_fen=$( (echo -e "position fen $current_fen moves $sf_move\nd\nquit") | stockfish | grep "Fen: " | cut -d ' ' -f 2-7)

    echo "Jugada realizada: $sf_move"
done

echo "Partida finalizada. Revisa $OUTPUT"
