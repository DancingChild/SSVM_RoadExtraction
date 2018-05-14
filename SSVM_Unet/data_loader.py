# -*- coding: utf-8 -*-
import time
from queue import Queue as th_queue
from thread_pool import thread_pool

class data_loader:

    def __init__(self, n_workers, load_func, max_interval=15, min_interval=5e-2):
        self._n_workers = n_workers
        self._thread_pool = thread_pool(n_workers + 1)
        self._task_queue = th_queue()
        self._oqueue = th_queue()
        self._closed = th_queue()
        self._max_interval = max_interval
        self._min_interval = min_interval
        def func_wrapper(*args, **kwargs):
            self._oqueue.put(load_func(*args, **kwargs))
        self._func_wrapper = func_wrapper
        self._thread_pool.add_task(self._manage)

    def _get(self, *args, **kwargs):
        """
        不保证结果与调用的顺序性，用于随机数据加载
        """
        osize = self._oqueue.qsize()
        self._task_queue.put((args, kwargs, osize))
        return self._oqueue.get()

    def __enter__(self):
        return self._get

    def __exit__(self, *args, **kwargs):
        self._closed.put(1)
        self._thread_pool.wait_completion()

    def _manage(self):
        interval = 1.0
        n_new_tasks = 0.0
        osize = 0
        while self._closed.qsize() == 0:
            n_wait_tasks = self._thread_pool.tasks.qsize()
            tps = (n_new_tasks - n_wait_tasks) / interval
            if n_wait_tasks > 0:
                interval *= (1.0 + 0.2 * n_wait_tasks/self._n_workers)
                if interval > self._max_interval:
                    interval = self._max_interval
            elif osize < self._n_workers:
                interval *= 0.6
                if interval < self._min_interval:
                    interval = self._min_interval
            n_new_tasks = self._task_queue.qsize()
            n_added_tasks = 0
            for i in range(n_new_tasks):
                args, kwargs, osize = self._task_queue.get()
                for j in range(1 + self._n_workers//(n_wait_tasks + osize + 1)):
                    self._thread_pool.add_task(self._func_wrapper, *args, **kwargs)
                    n_added_tasks += 1
            if tps > 0:
                print("<%.5f s> \033[1;33m%.4f t/s\033[0m \033[1;31m!%d\033[0m \033[1;32m+%d/%d\033[0m \033[1;34m%d\033[0m"%(interval, tps, n_wait_tasks, n_added_tasks, n_new_tasks, osize))
            n_new_tasks = n_added_tasks + n_wait_tasks #这里n_new_tasks变成总等待任务数量
            time.sleep(interval)
        print("\033[1;31mCLOSING......\033[0m")


if __name__ == "__main__":

    import time

    def func(s, interval):
        time.sleep(interval)
        return s

    with data_loader(n_workers=3, load_func=func) as data:
        for i in range(1000):
            d = data(i, 0.5)
            print("inx: %d | ret: %d"%(i, d))
            if i > 150 and i < 400:
                time.sleep(0.1)
            else:
                time.sleep(0.2)

