import unittest
import main
import dirgnn

class TestDataPermanence(unittest.TestCase):

    def setUp(self):
        self.graph_1 = main.run(gen_node_deltas=True, update_task_baseline_deltas=True)
        self.graph_2 = main.run(gen_node_deltas=False, update_task_baseline_deltas=False)

    def test_baseline_deltas(self):
        for node in self.graph_1:
            self.assertEqual(self.graph_1._node[node]['baseline_delta'], self.graph_2._node[node]['baseline_delta'])

    def test_edge_equality(self):
        self.assertEqual(self.graph_1.edges(),self.graph_2.edges())

    def test_failure(self):
        graph_thats_not_the_same = main.run(gen_node_deltas=True, update_task_baseline_deltas=False)

        for node in self.graph_1:
            self.assertEqual(self.graph_1._node[node]['baseline_delta'], graph_thats_not_the_same._node[node]['baseline_delta'])


class TestGraphFunctions(unittest.TestCase):

    def setUp(self):
        self.DAG = graph_1 = main.run(gen_node_deltas=False, update_task_baseline_deltas=False)

    def test_topological_sort_with_random_priority(self):
        # TODO: implement
        pass

if __name__ == '__main__':
    unittest.main()

