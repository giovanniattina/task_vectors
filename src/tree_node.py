class ThreeNode:
    def __init__(self, operation):
        self.operation = operation
        self.accuracy = None
        self.edges = []

    def add_edge(self, node):
        if node not in self.edges:
            self.edges.append(node)

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def __repr__(self):
        return f"ThreeNode({self.operation}, {self.accuracy})"