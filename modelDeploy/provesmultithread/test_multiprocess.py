# main.py
import multiprocessing as mp
#import uvicorn
from api import create_api
from inference_server import inference_server
#from mcts_worker import mcts_worker
#from gradio_app import launch_gradio

def mcts_worker(id, mcts_task_q, inference_q, inference_response_q, mcts_result_q):
    print("MCTS worker started", id)
    for _ in range(10):
        inference_q.put((str(id)+"_task"+str(_), None))
        response = inference_response_q.get()
        print(f"MCTS worker {id} received response: {response}")
    print("MCTS worker finished", id)
    

def main():
    print("Main process started")

    inference_q = mp.Queue()
    inference_response_q = mp.Queue()
    mcts_task_q = mp.Queue()
    mcts_result_q = mp.Queue()

    gpu = mp.Process(
        target=inference_server,
        args=(inference_q, inference_response_q),
    )
    gpu.start()

    api_app = create_api(mcts_task_q)
    api_proc = mp.Process(
        target=uvicorn.run,
        args=(api_app,),
        kwargs={"host": "0.0.0.0", "port": 8000},
    )
    api_proc.start()
    api_proc.join()

    workers = [
        mp.Process(
            target=mcts_worker,
            args=(id, mcts_task_q, inference_q, inference_response_q, mcts_result_q),
        )
        for id in range(mp.cpu_count())
    ]

    
    for w in workers:
        w.start()
    gpu.join()   # ← THIS IS REQUIRED

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    mp.freeze_support()
    main()



    """

"""
    #gpu.terminate()
    #for w in workers:
    #    w.terminate()