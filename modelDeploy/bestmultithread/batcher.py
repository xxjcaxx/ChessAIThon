import time
import queue



avg_batch_size = 0.0
batch_count = 0
batch_timeout_counter = 0


def batcher_loop(batcher_q, worker_response_queues, inference_q, inference_response_q):
    global avg_batch_size, batch_count, batch_timeout_counter
    print("Batcher started")
    batch = []
    BATCH_SIZE = 32
    GET_TIMEOUT = 0.005 
    batch_start = None 
    TIMEOUT = 0.02  # 20 ms
    last_flush = time.time()

    while True:
        try:
            item = batcher_q.get(timeout=GET_TIMEOUT)
            #print("Batcher received item from worker:", item)
            if not batch:
                batch_start = time.time()
            batch.append(item)
        except queue.Empty:
            pass


        # Si batch completo
        if len(batch) >= BATCH_SIZE:
            process_batch(batch, worker_response_queues, inference_q, inference_response_q)
            batch = []
            batch_start = None
            continue

        # Si pasó TIMEOUT y hay algo pendiente
        if batch and (time.time() - batch_start) >= TIMEOUT:
            process_batch(batch, worker_response_queues, inference_q, inference_response_q)
            batch = []
            #last_flush = time.time()
            batch_start = None
            batch_timeout_counter += 1

def process_batch(batch, worker_response_queues, inference_q, inference_response_q):
    #inputs = [x[1] for x in batch]
    #global avg_batch_size, batch_count, batch_timeout_counter
    #batch_count += 1
    #avg_batch_size += (len(batch) - avg_batch_size) / batch_count
    #print("process batch",batch,len(batch), avg_batch_size, batch_timeout_counter, batch_count)
    #print("Batcher processing batch of size:", len(batch))
    inference_q.put(batch)

    predictions = inference_response_q.get()
    #print(predictions)
    outputs = predictions  # [((id_worker, (id_thread, fen) ), move)]

    for (id_worker, _), out in zip(batch, outputs):
        worker_response_queues[id_worker].put(out)
