import math


class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    @property
    def is_empty(self):
        return not bool(self.items)


class PriorityQueue:
    def __init__(self):
        self.low_priority = []
        self.high_priority = []

    def enqueue(self, item, priority=False):
        if priority:
            self.high_priority.insert(0, item)
        else:
            self.low_priority.insert(0, item)

    def dequeue(self):
        if bool(self.high_priority):
            return self.high_priority.pop()

        return self.low_priority.pop()

    def peek(self):
        if bool(self.high_priority):
            return self.high_priority[-1]

        return self.low_priority[-1]

    @property
    def length(self):
        return len(self.high_priority) + len(self.low_priority)

    @property
    def is_empty(self):
        return not (bool(self.high_priority) or bool(self.low_priority))


class Stack:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def peek(self):
        return self.stack[-1]

    def length(self):
        return len(self.stack)

    @property
    def is_empty(self):
        return not bool(self.stack)


class LinkedListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return self.data


class LinkedList:
    def __init__(self, nodes=None):
        self.head = None
        if nodes:
            node = LinkedListNode(nodes.pop(0))
            self.head = node
            for elem in nodes:
                node.next = LinkedListNode(elem)
                node = node.next

    def __getitem__(self, val):
        if isinstance(val, int):
            if val < 0:
                node = self.head
                nodes = []
                while node is not None:
                    nodes.append(node)
                    node = node.next

                return nodes[val]

            for i, v in enumerate(self):
                if i == val:
                    return v

            raise IndexError('list index out of range')

        if isinstance(val, slice):
            nodes = []
            for i, v in enumerate(self):
                if val.start and val.stop and i >= val.start and i < val.stop:
                    if val.step:
                        if i % val.step == 0:
                            nodes.append(v)
                        continue

                    nodes.append(v)
                    continue

                if not val.stop and i >= val.start:
                    if val.step:
                        if i % val.step == 0:
                            nodes.append(v)
                        continue

                    nodes.append(v)
                    continue

                if not val.start and val.stop and i < val.stop:
                    if val.step:
                        if i % val.step == 0:
                            nodes.append(v)
                        continue

                    nodes.append(v)
                    continue

            return nodes

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(node.data)
            node = node.next
        nodes.append('None')
        return ' -> '.join(nodes)

    def __iter__(self):
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def __len__(self):
        if self.head is None:
            return 0

        for i, _ in enumerate(self):
            pass

        return i + 1

    def append(self, data):
        node = LinkedListNode(data)

        if self.head is None:
            self.head = node
            return

        for curr_node in self:
            pass

        curr_node.next = node

    def insert(self, index, data):
        node = LinkedListNode(data)

        if index == 0:
            node.next = self.head
            self.head = node
            return

        if self.head is None:
            raise RuntimeError('missing head')

        prev = self.head
        for i, elem in enumerate(self):
            if i == index:
                prev.next = node
                node.next = elem
                return
            prev = elem

        raise RuntimeError('unable to insert')

    def pop(self, index=None):
        if index == 0:
            self.head = self.head.next
            return

        if isinstance(index, int):
            prev = self.head
            for i, elem in enumerate(self):
                if i == index:
                    prev.next = elem.next
                    return
                prev = elem

            raise IndexError('list index out of range')

        prev = self.head
        if index is None:
            for elem in self:
                if elem.next is None:
                    if elem == self.head:
                        self.head = None
                        return
                    prev.next = None
                    return
                prev = elem

            raise IndexError('pop from empty list')

        raise RuntimeError('unable to pop')

    def reverse(self):
        if self.head is None:
            raise RuntimeError('missing head')

        prev = None
        node = self.head
        tail = node.next

        while node is not None:
            node.next = prev
            prev = node
            node = tail
            if tail:
                tail = tail.next

        self.head = prev

        return self


class LinkedListQueue:
    def __init__(self, nodes=None):
        self.items = LinkedList([n for n in nodes if nodes])

    def __len__(self):
        return len(self.items)

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    @property
    def is_empty(self):
        return not len(self.items)


class GraphNode:
    def __init__(self, key):
        self.key = key
        self.neighbors = []

    def add_neighbor(self, node):
        self.neighbors.append(node)


class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.edges = []
        self.nodes = []
        self.queue = Queue()

    def add_edge(self, node_1_key, node_2_key):
        node_1 = self.get_node(node_1_key)
        node_2 = self.get_node(node_2_key)

        node_1.add_neighbor(node_2)
        self.edges.append(f'{node_1_key}::{node_2_key}')

        if not self.directed:
            node_2.add_neighbor(node_1)

    def add_node(self, key):
        self.nodes.append(GraphNode(key)),

    def get_node(self, key):
        return next((x for x in self.nodes if x.key == key), None)

    def show(self, node):
        key = node.key
        neighbors = node.neighbors

        res = key

        if len(neighbors):
            for neighbor in neighbors:
                res += f'\n{key} => {neighbor.key}'

        return res

    def log(self):
        return print('\n'.join(map(self.show, self.nodes)))

    def breadth_first_search(self, key, visit_fn):
        """Visit closest nodes first and branch out from there."""
        self.queue.enqueue(self.get_node(key))
        visited = {n.key: False for n in self.nodes}

        while not self.queue.is_empty:
            curr_node = self.queue.dequeue()

            if not visited.get(curr_node.key):
                visit_fn(curr_node)
                visited[curr_node.key] = True

            for neighbor in curr_node.neighbors:
                if not visited.get(neighbor.key):
                    self.queue.enqueue(neighbor)

    def depth_first_search(self, key, visit_fn):
        """Visit deepest nodes first and walk back up from there."""
        visited = {n.key: False for n in self.nodes}

        def explore(node):
            if visited.get(node.key):
                return

            visit_fn(node)
            visited[node.key] = True

            for neighbor in node.neighbors:
                if not visited.get(neighbor.key):
                    explore(neighbor)

        explore(self.get_node(key))


class TreeNode:
    def __init__(self, key):
        self.children = []
        self.key = key

    def add_child(self, key):
        child = TreeNode(key)
        self.children.append(child)

        return child


class Tree:
    def __init__(self, key):
        self.root = self.add_node(key)
        self.res = ''

    def add_node(self, key):
        return TreeNode(key)

    def print(self):
        def traverse(node, visit_fn, depth):
            visit_fn(node, depth)

            for child in node.children:
                traverse(child, visit_fn, depth + 1)

        def add_key_to_res(node, depth):
            if depth == 1:
                print(node.key)
            else:
                print(f'{"  " * (depth - 1)}{node.key}')

        traverse(self.root, add_key_to_res, 1)


def bubble_sort(to_sort):
    swap = True

    while swap:
        swap = False

        for index, item in enumerate(to_sort):
            next_item = (
                to_sort[index + 1] if (index + 1) < len(to_sort) else None)

            if next_item and next_item < item:
                print(to_sort)
                to_sort.insert(index, to_sort.pop(index + 1))
                swap = True

                continue

    return to_sort


def insert_sort(to_sort):
    n = len(to_sort)

    for i in range(1, n):
        print(to_sort)

        curr = to_sort[i]
        j = i - 1

        while j >= 0 and curr < to_sort[j]:
            to_sort[j + 1] = to_sort[j]
            j -= 1
        to_sort[j + 1] = curr

    print(to_sort)
    return to_sort


def merge_sort(to_sort):
    if len(to_sort) < 2:
        return to_sort

    middle = math.floor(len(to_sort) / 2)
    left = to_sort[0:middle]
    right = to_sort[middle:]

    if isinstance(left, int):
        left = [left]

    if isinstance(right, int):
        right = [right]

    return merge(merge_sort(left), merge_sort(right))


def merge(left, right):
    sort = []

    while len(left) and len(right):
        if left[0] <= right[0]:
            sort.append(left.pop(0))
        else:
            sort.append(right.pop(0))

    res = sort + left + right
    print(res)

    return res
