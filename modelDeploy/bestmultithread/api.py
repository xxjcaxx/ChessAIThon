# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import multiprocessing as mp
import uuid
import json

from fastapi.middleware.cors import CORSMiddleware



def create_api(task_q, tasks_result_q):
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # en producción pon tu dominio
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class Req(BaseModel):
        fen: str
        simulations: int
        mcts_tree: bool = False

    @app.post("/predict")
    def predict(req: Req):
        print("\033[1;36m♟️  Request received\033[0m | sims="+str(req.simulations)+"")
        print("\033[36mFEN:\033[0m "+str(req.fen)+"\n")

        task_id = uuid.uuid4().hex
        task_q.put((task_id, req.fen, req.simulations, req.mcts_tree))

        while True:
            result = tasks_result_q.get()
            if len(result) == 5:
                rid, move, total_visits, alternatives, mcts_tree_json = result
            else:
                rid, move, total_visits, alternatives = result
                mcts_tree_json = None
           
            if rid == task_id:
                response = {"move": move, "visits": total_visits, "alternatives": alternatives}
                if req.mcts_tree:
                    if isinstance(mcts_tree_json, str) and mcts_tree_json:
                        try:
                            response["mcts_tree"] = json.loads(mcts_tree_json)
                        except json.JSONDecodeError:
                            response["mcts_tree"] = None
                    else:
                        response["mcts_tree"] = mcts_tree_json
                return response

    return app
