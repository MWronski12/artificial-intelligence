class Stats:
    def __init__(self) -> None:
        self.nodes_visited: int = 0
        self.nodes_pruned: int = 0
        self.leaf_count: int = 0
        self.aggregated_depth: int = 0
        self.execution_time: float = 0.0

    def avg_depth(self) -> int:
        return self.aggregated_depth / self.leaf_count

    def print(self):
        print("============ STATS ===========")
        print("Execution time:", self.execution_time, "s")
        print("Nodes visited:", self.nodes_visited)
        print("Average tree depth:", self.avg_depth())
        print("Nodes pruned:", self.nodes_pruned)
        print("==============================")
