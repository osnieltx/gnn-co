import multiprocessing
import time
import random


def worker(counter, lock, max_count):
    while True:
        with lock:
            if counter.value >= max_count:
                return
            counter.value += 1
            current = counter.value
        print(f"Process {multiprocessing.current_process().name} increased counter to {current}")
        time.sleep(random.uniform(0.1, 0.5))


def main():
    max_count = 10
    num_processes = 3

    counter = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    processes = []

    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker, args=(counter, lock, max_count))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Final counter value:", counter.value)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Optional, default on macOS/Windows
    main()