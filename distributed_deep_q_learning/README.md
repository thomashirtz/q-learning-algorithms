# Distributed Deep Q Learning

## Basic building blocks

I will first introduce the two basic building blocks to realize the distributed deep Q-learnign algorithm
namely: shared variable for the global counter and the Hogwild! network update.

### Shared variable

We want to generate a global counter that will be incremented each time one of the process
do an iteration.

This is the minimal code that can achieve that:

```
import torch.multiprocessing as mp


class Worker:
    def run(self, global_step):
        for i in range(10):
            with global_step.get_lock():
                global_step.value += 1

if __name__ == '__main__':
    global_step = mp.Value('i', 0)
    num_processes = 4
    processes = []

    for rank in range(num_processes):
        p = mp.Process(target=Worker().run, args=(global_step,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print(global_step.value)
```

** Note: **
* Multiprocessing does not work properly on notebooks. There is some information on [[stackoverflow]](https://stackoverflow.com/questions/48846085/python-multiprocessing-within-jupyter-notebook) 
if someone wants to do some further testing. I generally try to tinker on jupyter notebook and clean the code into an
IDE such as Pycharm. I struggled some time before realizing that the reason why my tinkering was not working was because 
of the environment I was working on.
* when working with multiprocessing `if __name__ == '__main__':` is mandatory. I quote [[this]]() thread
    >The `multiprocessing` module works by creating new Python processes that will import your module. If you did not add `__name__== '__main__'` protection then you would enter a never ending loop of new process creation.
* `mp.Value('i', 0)` allows to share a variable between all the process. the `i` represent the type of variable that is 
being stored (integer) and 0 is the initial value. 
* A locked is used to avoid issue with the update of this variable:
    ```
    with global_step.get_lock():
                    global_step.value += 1
    ```
    Here is a [[thread]](https://stackoverflow.com/questions/2080660/python-multiprocessing-and-a-shared-counter) about more 
    information related to shared counter
* The arguments of the function needs to be put in the `args` argument of the Process function.

### Shared Network

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp


INPUT_DIMENSION = 10
OUTPUT_DIMENSION = 4


class Worker:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def run(self, num_steps):
        for i in range(num_steps):

            x = torch.ones((1, INPUT_DIMENSION))
            prediction = self.model(x)
            target = -torch.ones((1, OUTPUT_DIMENSION))

            loss = nn.MSELoss()(prediction, target)
            loss.backward()

            if i % 10 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()


if __name__ == '__main__':
    model = nn.Linear(INPUT_DIMENSION, OUTPUT_DIMENSION)
    model.share_memory()
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    num_processes = 4
    num_steps_per_worker = 40

    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=Worker(model, optimizer).run, args=(num_steps_per_worker,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print(model(torch.ones((1, INPUT_DIMENSION))).tolist())
```

This is a small runnable example of the Hogwild! update. It is a simplified version of the [[Pytorch tutorial]](https://pytorch.org/docs/stable/notes/multiprocessing.html)
`torch.multiprocessing` is an extension of the `multprocessing` module. It allows to use multiprocessing with the pytorch 
library, while conserving its other features.