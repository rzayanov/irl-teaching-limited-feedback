import time

timers = {}


def start_timer(name):
    timers[name] = time.time()


def print_timer(name):
    delta = int(time.time() - timers[name])
    res = delta_str(delta)
    print(f"{name} run time: {res}")


def delta_str(delta):
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        res = f"{hours}h {minutes:02d}m {seconds:02d}s"
    elif minutes:
        res = f"{minutes}m {seconds:02d}s"
    else:
        res = f"{seconds}s"
    return res
