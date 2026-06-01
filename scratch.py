import numpy as np
import multiprocessing
import queue
import time

def _worker(q, s, m, n):
    try:
        # simulate
        time.sleep(1)
        res = np.zeros((n, len(s)))
        q.put(("success", res))
    except Exception as e:
        q.put(("error", e))

if __name__ == "__main__":
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    # Big enough array to fill pipe
    s = np.arange(1000000)
    p = ctx.Process(target=_worker, args=(q, s, "EMD", 5))
    p.start()
    
    res_tuple = None
    aborted = False
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > 2.0: # don't abort in this test
            pass
            
        try:
            res_tuple = q.get(timeout=0.1)
            break
        except queue.Empty:
            if not p.is_alive():
                try:
                    res_tuple = q.get(timeout=0.1)
                except queue.Empty:
                    pass
                break
                
    p.join()
    if res_tuple:
        print("Got result size:", res_tuple[1].shape)
    else:
        print("Failed.")

