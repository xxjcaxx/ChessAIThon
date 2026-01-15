import time
import queue



avg_batch_size = 0.0
batch_count = 0
batch_timeout_counter = 0


def batcher_loop(batcher_q, worker_response_queues):
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
            if not batch:
                batch_start = time.time()
            batch.append(item)
        except queue.Empty:
            pass

        # Si batch completo
        if len(batch) >= BATCH_SIZE:
            process_batch(batch, worker_response_queues)
            batch = []
            batch_start = None
            continue

        # Si pasó TIMEOUT y hay algo pendiente
        if batch and (time.time() - batch_start) >= TIMEOUT:
            process_batch(batch, worker_response_queues)
            batch = []
            #last_flush = time.time()
            batch_start = None
            batch_timeout_counter += 1

def process_batch(batch, worker_response_queues):
    inputs = [x[1] for x in batch]
    global avg_batch_size, batch_count, batch_timeout_counter
    batch_count += 1
    avg_batch_size += (len(batch) - avg_batch_size) / batch_count
    print("process batch",batch,len(batch), avg_batch_size, batch_timeout_counter, batch_count)
    
    #outputs = inputs #model_infer(inputs)  # UNA llamada al modelo

    #for (id_worker, _), out in zip(batch, outputs):
    #    worker_response_queues[id_worker].put(out)
