import time
import queue
import torch



def batcher_loop(batcher_q, worker_response_queues,
                 inference_q, inference_response_q, last_batch_avg):

    print("Batcher started")

    BATCH_SIZE = 50
    TIMEOUT = 0.04  # 40 ms

    avg_batch_size = 0.0
    batch_count = 0
    batch_timeout_counter = 0

    batch = []
    batch_start = None

    while True:
        try:
            if not batch:
                # Primer elemento: esperamos bloqueando
                item = batcher_q.get()
                batch = [item]
                batch_start = time.perf_counter()
                continue

            remaining = TIMEOUT - (time.perf_counter() - batch_start)
            if remaining <= 0:
                raise queue.Empty

            item = batcher_q.get(timeout=remaining)
            batch.append(item)

        except queue.Empty:
            # Timeout vencido
            if batch:
                batch_size = len(batch)
                process_batch(batch, worker_response_queues,
                              inference_q, inference_response_q)

                batch_count += 1
                avg_batch_size += (batch_size - avg_batch_size) / batch_count
                batch_timeout_counter += 1

                batch = []
                batch_start = None

        # Batch completo
        if len(batch) >= BATCH_SIZE:
            batch_size = len(batch)
            process_batch(batch, worker_response_queues,
                          inference_q, inference_response_q)

            batch_count += 1
            avg_batch_size += (batch_size - avg_batch_size) / batch_count

            batch = []
            batch_start = None

        # Publicar stats (barato)
        last_batch_avg[0] = avg_batch_size
        last_batch_avg[1] = batch_timeout_counter
        last_batch_avg[2] = batch_count
        if avg_batch_size < BATCH_SIZE * 0.7:
            BATCH_SIZE = max(16, BATCH_SIZE - 8)
        elif avg_batch_size > BATCH_SIZE * 0.9:
            BATCH_SIZE = min(128, BATCH_SIZE + 8)



def process_batch(batch, worker_response_queues, inference_q, inference_response_q):

    inference_q.put(batch)

    predictions = inference_response_q.get()
    #print(predictions)
    outputs = predictions  # [((id_worker, (id_thread, fen) ), move)]

    for (id_worker, _), out in zip(batch, outputs):
        worker_response_queues[id_worker].put(out)
