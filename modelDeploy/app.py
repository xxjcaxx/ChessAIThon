import gradio as gr
from chessmodel import predict_chess_move
from chessgame import chessmarro_mcts_predict_chess_move


# Create Gradio interface
iface = gr.Interface(
    fn=chessmarro_mcts_predict_chess_move,
     inputs=[
        gr.Textbox(label="FEN"), 
        gr.Slider(minimum=10, maximum=1000, step=10, value=10, label="Number of Simulations"),  # Slider para elegir simulaciones
       # gr.Checkbox(label="Show simulation details", value=False)
    ],
    outputs="text",
    title="Chess Move Predictor",
   # live=True  # Para mostrar los resultados en tiempo real
)

iface.launch(share=True)
