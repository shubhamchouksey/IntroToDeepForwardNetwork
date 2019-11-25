# IntroToDeepForwardNetwork

**x1-x2** is just a function where its value to be either positive, negetive or zero. The truth table of the x1-x2 looks like this:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 4 | 4 | 0|
| 2 | 4 | -1 |
| 4 | 2 | 1 |
| 2 | 2 | 0 |

In words, the (x1-x2) function is 0 if both inputs are the same, or +ve if x1>x2(+1) or -ve if x1<x2(-1). 

****************************
**Exercise 2:** In the python file SimpleDeepForwardNetwork.py, you will create code for a neural network which performs the (x1-x2) operation. The following elements are required:

1. An input weight tensor
2. A hidden layer with 10 units using sigmoid activation
3. A output weight tensor
4. An output unit with sigmoid activation

There are 4 possible test (input) cases. Test your code for all cases.  
