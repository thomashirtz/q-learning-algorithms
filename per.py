import numpy as np
from itertools import tee


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class SumTree:
    def __init__(self, capacity):
        self.data_pointer = 0
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)
        self.tree = np.zeros(2 * capacity - 1)

    def push(self, value, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self._update_tree(tree_index, value)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, indexes, values):
        for index, value in zip(indexes, values):
            # tree_index = self.data_pointer + self.capacity - 1
            self._update_tree(index, value)

    def get(self, s):
        index = self._retrieve(0, s)
        index_data = index - self.capacity + 1
        return index, self.tree[index], self.data[index_data]

    def _retrieve(self, index, value):
        index_left_child = 2 * index + 1
        index_right_child = index_left_child + 1

        if index_left_child >= len(self.tree):
            return index
        if value <= self.tree[index_left_child]:
            return self._retrieve(index_left_child, value)
        else:
            return self._retrieve(index_right_child, value - self.tree[index_left_child])

    def _update_tree(self, index, value):
        difference = value - self.tree[index]
        self.tree[index] = value
        while index > 0:
            index = (index - 1) // 2
            self.tree[index] += difference

    @property
    def total(self):
        return self.tree[0]

    @property
    def max_leaf_value(self):
        return np.max(self.tree[-self.capacity:])

    @property
    def min_leaf_value(self):
        return np.min(self.tree[-self.capacity:][np.nonzero(self.tree[-self.capacity:])])

    def __len__(self):
        return np.count_nonzero(self.tree[-self.capacity:])

alpha = 0.6
beta = 0.4
epsilon = 0.01

class ProportionalPrioritizedMemory:
    def __init__(self, capacity, epsilon=0.01):
        self.epsilon = epsilon
        self.capacity = capacity
        self.maximum_priority = 1.0
        self.tree = SumTree(capacity)

    def push(self, experience):
        priority = self.tree.max_leaf_value if self.tree.max_leaf_value else self.maximum_priority
        self.tree.push(value=priority, data=experience)

    def sample(self, batch_size, beta=0.4):
        segments = np.linspace(0, self.tree.total, num=batch_size + 1)
        tuples = []
        for start, end in pairwise(segments):
            value = np.random.uniform(start, end)
            tuples.append(self.tree.get(value))
        indexes, priorities, experiences = zip(*tuples)

        probabilities = np.array(priorities) / self.tree.total
        maximum_weight = (self.capacity * self.tree.min_leaf_value / self.tree.total) ** (-beta)
        weights = np.power(self.capacity * probabilities, -beta) / maximum_weight

        return list(indexes), list(weights), list(experiences)

    def update(self, indexes, deltas, alpha=0.6):
        priorities = np.power(np.abs(deltas) + self.epsilon, alpha)
        clipped_priorities = np.clip(priorities, 0, self.maximum_priority)
        self.tree.update(indexes, clipped_priorities)

    def __len__(self):
        return len(self.tree)