import os
import threading
import time


_round = round


class KVFormator(object):
    def __init__(self):
        self.key_str_len = {}

    def __call__(self, key, value):
        value = str(value)
        l = self.key_str_len[key] = max(self.key_str_len.get(key, 0), len(value))
        return f"{key}={value}{' ' * (l - len(value))}"


class Timer(object):
    """
    1. 统计一段代码的运行的瞬时时间、平均时间、最大时间、最小时间
    2. 为了多进程多线程调试，打印类当前的进程id和线程id
    3.
    """
    def __init__(self):
        self.tic_data, self.total_cost_time, self.max_cost_time, self.min_cost_time = {}, {}, {}, {}
        self.count = {}
        self.formator = KVFormator()

    def tic(self, key):
        self.tic_data[key] = time.time()

    def toc(self, key, condition_func=None, round_num=3):
        if condition_func is not None and not condition_func(self):
            return
        self.count[key] = self.count.get(key, 0) + 1
        t = time.time() - self.tic_data[key]  # + self.data.get(key, 0)
        self.total_cost_time[key] = self.total_cost_time.get(key, 0) + t
        self.max_cost_time[key] = max(self.total_cost_time.get(key, 0), t)
        self.min_cost_time[key] = min(self.total_cost_time[key], t) if key in self.min_cost_time else t

        r = round_num
        print(f'{self.formator("tag", key)},'
              f' {self.formator("cur", _round(t, r))}s,'
              f' {self.formator("avg", _round(self.total_cost_time[key] / self.count[key], r))}s,'
              f' {self.formator("max", _round(self.max_cost_time[key], r))}s,'
              f' {self.formator("min", _round(self.min_cost_time[key], r))}s,'
              f' (pid={os.getpid()}, tid={threading.currentThread( ).ident})')
        return self.total_cost_time[key]


timer = Timer()


def timer_d(func):
    def wrapper(*args, **kwargs):
        timer.tic(func.__name__)
        res = func(*args, **kwargs)
        timer.toc(func.__name__)
        return res
    return wrapper


if __name__ == '__main__':
    t = Timer()
    for i in range(10):
        t.tic("sleep 0.3")
        time.sleep(0.3)
        t.toc("sleep 0.3")

    print("timer_d")

    @timer_d
    def f(t):
        time.sleep(t)

    for i in range(10):
        f(0.3)


