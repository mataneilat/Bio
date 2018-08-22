
from heapq import *
import itertools

class SimpleHeapIterator:

    def __init__(self, heap):
        self.heap = heap.copy()

    def __next__(self):
        if self.heap.is_empty():
            raise StopIteration
        priority, element = self.heap.pop_element()
        if element is None:
            raise StopIteration
      #  print("ELEM: " + element.__repr__() )
        return priority, element


class SimpleHeap:

    def __init__(self, elements=[], is_max = False):
        self.pq = []
        self.entry_finder = {}
        self.is_max = is_max
        self.counter = itertools.count()
        self.setup(elements)
        self.removed_element = -1

    def setup(self, elements):
        for element in elements:
            self.add_element(element)

    def add_element(self, element, priority = 0):
        if element in self.entry_finder:
            self.remove_element(element)

        count = next(self.counter)
        entry = [self.correct_priority(priority), count, element]
        self.entry_finder[element] = entry
       # print("ADDING:" + entry.__repr__())
        heappush(self.pq, entry)


    def reset_priority(self):
        for priority, element in self:
            if element != self.removed_element:
                self.add_element(element, priority=0)

    def remove_element(self, element):
        entry = self.entry_finder.pop(element)
        entry[-1] = self.removed_element

    def is_empty(self):
        return len(self) == 0

    def __iter__(self):
        return SimpleHeapIterator(self)

    def __len__(self):
        return len(self.entry_finder)

    def __repr__(self):
        return self.pq.__repr__()

    def copy(self):
        copy_heap = SimpleHeap(is_max=self.is_max)
        for priority, count, x in self.pq:
            if x != self.removed_element:
                copy_heap.add_element(x, priority=self.correct_priority(priority))
        return copy_heap

    def correct_priority(self, priority):
        return -priority if self.is_max else priority

    def pop_element(self):
        print(self.pq)
        while self.pq:
            priority, count, element = heappop(self.pq)
            if element != self.removed_element:
                del self.entry_finder[element]
                return self.correct_priority(priority), element
        return 0, None