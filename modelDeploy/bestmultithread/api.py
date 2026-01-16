# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import multiprocessing as mp
import uuid

def create_api(task_q, tasks_result_q):
    app = FastAPI()

    class Req(BaseModel):
        fen: str
        simulations: int

    @app.post("/predict")
    def predict(req: Req):
        print("Received request:", req)
        task_id = uuid.uuid4().hex
        task_q.put((task_id, req.fen, req.simulations))

        while True:
            rid, move = tasks_result_q.get()
            print("Checking result:", rid, move)
            if rid == task_id:
                return {"move": move}

    return app
