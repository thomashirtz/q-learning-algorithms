```
import time
import random
from collections import deque
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager

T = 20
B = 5
REPLAY_MINIMUM_SIZE = 10
REPLAY_MAXIMUM_SIZE = 100


class Actor:
    def __init__(self, global_buffer, rank):
        self.rank = rank
        self.local_buffer = []
        self.global_buffer = global_buffer

    def run(self, num_steps):
        for step in range(num_steps):
            data = f'{self.rank}_{step}'
            self.local_buffer.append(data)

            if len(self.local_buffer) >= B:
                self.global_buffer.put(self.local_buffer)
                self.local_buffer = []


class Learner:
    def __init__(self, replay):
        self.replay = replay

    def run(self, num_steps):
        while self.replay.size() <= REPLAY_MINIMUM_SIZE:
            time.sleep(0.1)
        for step in range(num_steps):
            batch = self.replay.sample(B)
            print(batch)

class Replay:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, experiences):
        self.memory.extend(experiences)

    def sample(self, n):
        return random.sample(self.memory, n)

    def size(self):
        return len(self.memory)


def send_data_to_replay(global_buffer, replay):
    while True:
        if not global_buffer.empty():
            batch = global_buffer.get()
            replay.push(batch)


if __name__ == '__main__':
    num_actors = 2

    global_buffer = mp.Queue()

    BaseManager.register("ReplayMemory", Replay)
    Manager = BaseManager()
    Manager.start()
    replay = Manager.ReplayMemory(REPLAY_MAXIMUM_SIZE)

    learner = Learner(replay)
    learner_process = mp.Process(target=learner.run, args=(T,))
    learner_process.start()

    actor_processes = []
    for rank in range(num_actors):
        p = mp.Process(target=Actor(global_buffer, rank).run, args=(T,))
        p.start()
        actor_processes.append(p)

    replay_process = mp.Process(target=send_data_to_replay, args=(global_buffer, replay,))
    replay_process.start()

    learner_process.join()
    [actor_process.join() for actor_process in actor_processes]
    replay_process.join()
```