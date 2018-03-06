import unittest

class TestDataType(unittest.TestCase):
    
    def test_int_input(self):
        data = np.array([[0,np.nan],
               [0,np.nan],
               [1,np.nan],
               [1,np.nan]])
        A = DiscreteDistribution({0:0.4, 1:0.6})
        B = ConditionalProbabilityTable([
            [0, 0, 1.0],
            [0, 1, 0.0],
            [1, 0, 0.0],
            [1, 1, 1.0]], [A] )
        
        s_A = State(A, 'A')
        s_B = State(B, 'B')
        
        model = BayesianNetwork('copy')
        model.add_states(s_A, s_B)
        model.add_transition(s_A, s_B)
        model.bake()
        
        new_data = em_bayesnet(model, data, 1)
        
        correct_output = np.array([
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1]])
        
        self.assertEqual(new_data.tolist(), correct_output.tolist())
        
    def test_str_input(self):
        data = np.array([['0',np.nan],
               ['0',np.nan],
               ['1',np.nan],
               ['1',np.nan]])
        A = DiscreteDistribution({'0':0.4, '1':0.6})
        B = ConditionalProbabilityTable([
            ['0', '0', 1.0],
            ['0', '1', 0.0],
            ['1', '0', 0.0],
            ['1', '1', 1.0]], [A] )
        
        s_A = State(A, 'A')
        s_B = State(B, 'B')
        
        model = BayesianNetwork('copy')
        model.add_states(s_A, s_B)
        model.add_transition(s_A, s_B)
        model.bake()
        
        new_data = em_bayesnet(model, data, 1)
        
        correct_output = np.array([
            ['0', '0'],
            ['0', '0'],
            ['1', '1'],
            ['1', '1']])
        
        self.assertEqual(new_data.tolist(), correct_output.tolist())
        

if __name__ == "__main__":
    unittest.main()
