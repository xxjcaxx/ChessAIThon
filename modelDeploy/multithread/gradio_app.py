# gradio_app.py
import gradio as gr
import multiprocessing as mp
import uuid

def launch_gradio(task_q):
    
    def predict(fen, simulations):
        task_id = uuid.uuid4().hex
        task_q.put((task_id, fen, simulations))

        while True:
            rid, move = result_q.get()
            if rid == task_id:
                return move

    iface = gr.Interface(
        fn=predict,
        inputs=["text", "number"],
        outputs="text",
    )
    iface.launch(server_name="0.0.0.0", server_port=7860)
