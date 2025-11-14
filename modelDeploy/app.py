import gradio as gr
import multiprocessing as mp
from threading import Lock
from chessgamemultithread import chessmarro_mcts_predict_chess_move
from chessmodel import init_model

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

print("Iniciando la aplicación de despliegue del modelo (import).")

# -------- Modelo lazy con lock --------
_model = None
_device = None
_model_lock = Lock()

def get_model():
    global _model, _device
    if _model is None:
        with _model_lock:
            if _model is None:
                print("Inicializando modelo (solo una vez en este proceso)...")
                try:
                    _model, _device = init_model()
                except Exception as e:
                    print(f"Warning: no se pudo inicializar el modelo: {e}")
                    _model, _device = None, None
    return _model, _device

# -------- Wrapper de Gradio --------
def predict_fn(fen: str, simulations: int):
    print(f"Predict called with FEN={fen} simulations={simulations}")
    model, device = get_model()
    if model is None:
        return "Modelo no inicializado. Revisa los logs del servidor."
    try:
        move, tree_json = chessmarro_mcts_predict_chess_move(
            fen, simulations, model, device, num_workers=64
        )
        # Normalizar la salida: devolver siempre una cadena UCI.
        # La función MCTS puede devolver un objeto `chess.Move`; forzamos a str()
        # para que tanto Gradio (que muestra texto) como FastAPI (que serializa a JSON)
        # reciban el mismo formato consistente (por ejemplo 'e2e4').
        try:
            return str(move) if move is not None else None
        except Exception:
            # Fallback seguro: devuelve representación fallback
            return repr(move)
    except Exception as e:
        return f"Error al ejecutar MCTS: {e}"

# -------- Interfaz Gradio --------
iface = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.Textbox(label="FEN"),
        gr.Slider(minimum=1, maximum=2000, step=1, value=10, label="Number of Simulations"),
    ],
    outputs="text",
    title="Chess Move Predictor (multithreaded)",
)

# -------- FastAPI --------
app = FastAPI(title="Chess Move Predictor API")

class Request(BaseModel):
    fen: str
    simulations: int

@app.post("/predict")
def predict_api(req: Request):
    """Endpoint REST limpio para Postman/curl"""
    result = predict_fn(req.fen, req.simulations)
    # Asegurar que el valor que se serializa a JSON sea una cadena (UCI)
    try:
        move_str = str(result) if result is not None else None
    except Exception:
        move_str = repr(result)
    return {"move": move_str}

# -------- Lanzamiento combinado --------
def launch_gradio():
    iface.launch(
        share=True,               # URL pública
        server_name="0.0.0.0",    # API REST local accesible desde localhost
        server_port=7860           # Puerto de Gradio
    )

if __name__ == '__main__':
    # Multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Lanzar Gradio en un hilo para no bloquear FastAPI
    threading.Thread(target=launch_gradio, daemon=True).start()

    # Lanzar FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
