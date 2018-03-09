# Bayesian Network with Hidden Variables

bayesnet_em predicts values for a hidden variable in a Bayesian network by implementing the expectation maximization algorithm. It works as an extension to the Beysian network implementation in Pomegranade.

## Installation

### Dependencies
> - numpy 
> - pomegranade 

### Installing
If you have Git installed:
> `pip install git+https://github.com/nnvutisa/EM_BayesNet.git`

## Usage

A Bayesian network is a probabilistic graphical model that represents relationships between random variables as a direct acyclic graph. Each node in the network represents a random variable whearas each edge represents a conditional dependency. Bayesian networks provides an efficient way to construct a full joint probability distribution over the variables. The random varibles can either be observed variables or unobserved variables, in which case they are called hidden (or latent) variables. 

Pomegranade currently supports a discrete Baysian network. Each node represents a categorical variable, which means it can take on a discrete number of values. The model parameters can be learned from data. However (at least as of now), it does not support a network with hidden variables. The purpose of bayesnet_em is to work with the pomegranade Bayesian network model to predict values of the hidden variables.  

bayesnet_em takes an already constructed and initialized `BayesianNetwork` object, a data array, and the index of the hidden node, and returns a complete data set.

### Example
Let's use a simple example. Suppose there is a bag of fruits that contains apples and bananas. The observed variables for each sample taken from the bag are color, taste, and shape, while the label of the type of fruits was not recorded. We would like to predict the type of fruits for each sample along with the full probability distribution. This can be represented by a Bayesian network as:

This simple relationship describes a naive Bayes model. The full joint probability distribution is
> P(F,C,T,S) = P(F)\*P(C|F)\*P(T|F)\*P(S|F)

Start with building a Bayesian network model using pomegranade.

```
import numpy as np
from pomegranate import *
```

Here's our data. bayesnet_em suppost data types of int and string.

```
data = np.array([[np.nan, 'yellow', 'sweet', 'long'],
                [np.nan, 'green', 'sour', 'round'],
                [np.nan, 'green', 'sour', 'round'],
                [np.nan, 'yellow', 'sweet', 'long'],
                [np.nan, 'yellow', 'sweet', 'long'],
                [np.nan, 'green', 'sour', 'round'],
                [np.nan, 'green', 'sweet', 'long'],
                [np.nan, 'green', 'sweet', 'round']])
```

The columns represent the nodes in a specified order (fruit, color, taste, shape). The order of the columns have to match the order of the nodes (states) when constructing the Bayesian network. The first column with the `nan` values is the hidden node. Next, create the distributions of all the nodes and initialize the probabilities to some non-uniform values. The first node is just P(F). The other three nodes are described by conditional probabilities.

```
Fruit = DiscreteDistribution({'banana':0.4, 'apple':0.6})
Color = ConditionalProbabilityTable([['banana', 'yellow', 0.6],
                                 ['banana', 'green', 0.4],
                                 ['apple', 'yellow', 0.6],
                                 ['apple', 'green', 0.4]], [Fruit] ) 
Taste = ConditionalProbabilityTable([['banana', 'sweet', 0.6],
                                    ['banana', 'sour', 0.4],
                                    ['apple', 'sweet', 0.4],
                                    ['apple', 'sour', 0.6]], [Fruit])
Shape = ConditionalProbabilityTable([['banana', 'long', 0.6],
                                    ['banana', 'round', 0.4],
                                    ['apple', 'long', 0.4],
                                    ['apple', 'round', 0.6]], [Fruit])
```
Now, create the state (node) objects

```
s_fruit = State(Fruit, 'fruit')
s_color = State(Color, 'color')
s_taste = State(Taste, 'taste')
s_shape = State(Shape, 'shape')
```

and the `BayesianNetwork` object.

```
model = BayesianNetwork('fruit')
```

Add states and edges to the network.

```
model.add_states(s_fruit, s_color, s_taste, s_shape)
model.add_transition(s_fruit, s_color)
model.add_transition(s_fruit, s_taste)
model.add_transition(s_fruit, s_shape)
model.bake()
```

Now that we have the initialized model, we want to fill in the first column.

```
from bayesnet_em import *
```

call the `em_bayesnet` function. Our hidden node index is 0 since it is the first node.

```
hidden_node_index = 0
new_data = em_bayesnet(model, data, hidden_node_index)
```

The returned array is the filled in data. Note that the function does not modify the model or the input data; it only returns a new data array.

```
>>> new_data
array([['banana', 'yellow', 'sweet', 'long'],
       ['apple', 'green', 'sour', 'round'],
       ['apple', 'green', 'sour', 'round'],
       ['banana', 'yellow', 'sweet', 'long'],
       ['banana', 'yellow', 'sweet', 'long'],
       ['apple', 'green', 'sour', 'round'],
       ['banana', 'green', 'sweet', 'long'],
       ['apple', 'green', 'sweet', 'round']], 
      dtype='|S32')
```

Of course the function does not actually know the meaning of the labels. In real uses, it would be more appropriate to use integer labels or something like 'group1', 'group2', etc. For the sake of this example, I just assigned the correct label names.

If you want to furthur use the Bayesian network model (to see the distributions or make predictions for example), you can fit the model to the complete data set. 

```
new_model = model.fit(new_data)
```

The model can be used to answer questions like, "what is the probability that a banana is yellow?"

```
>>> new_model.predict_proba({'fruit':'banana'})[1].parameters
[{'green': 0.25000000000000017, 'yellow': 0.74999999999999989}]
```

What is the probability that a sample would be a sweet, round, and green apple?

```
>>> new_model.probability(['apple', 'green', 'sweet', 'round'])
0.12500000000000003
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
