## Week 4 - Deep Neural Networks(DNN)

Understand the key computations underlying deep learning, use them to build and train deep neural networks, and apply it to computer vision.

### I. DNN

#### 1. Deep L-layer neural network

![W4.1](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W4/W4.1.png?raw=true)

- Shallow NN is a NN with one or two layers.
- Deep NN is a NN with three or more layers.
- We will use the notation L to denote the number of layers in a NN.

eg: Four layer neural network with three hidden layers

![W4.2](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W4/W4.2.png?raw=true)

__Notation__:

We will use the notation `L` to denote the number of layers in a NN.

```
	  L  = # layers
	n[0] = # neurons in input layer. 
	n[L] = neurons in output layer.

	n[l] = # units in layer l
	a[l] = activation in layer l (a[0]=X input)
	a[0] = X
	z[L] = Y

	g[l] = the activation function
	w[l] = weights for Z[l]
	b[l] = biases for Z[l]
	
```

L=4 in this example:

```
	n[1]=5, n[2]=5, n[3]=3, n[4]=1
	n[0]=nx=3
```

#### 2. Forward Propagation in a Deep Network

![W4.3](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W4/W4.3.png?raw=true)

__General Rule__:

```
	z[l]=w[l]a[l-1]+b[l]
	a=g[l](z[l])

Vectorized:
	
	Z[l]=W[l]A[l-1]+b[l]
	A[l]=g[l](Z[l])
```

#### 3. Getting your matrix dimensions right

__General formula of matrix dimensions__

* Dimension of W is (n[l],n[l-1]) . Can be thought by right to left.
* Dimension of b is (n[l],1)
* dw has the same shape as W, while db is the same shape as b
* Dimension of Z[l], A[l], dZ[l], and dA[l] is (n[l],m)

#### 4. Why deep network works better than shallow neural network?

deep neural network = many hidden units

__A. simple ==> complex__

Deep NN makes relations with data from simpler to complex. In each layer it tries to make a relation with the previous layer. 

Eg:
- __Face recognition application__: Image ==> Edges ==> Face parts ==> Faces ==> desired face
- __Audio recognition application__: Audio ==> Low level sound features like (sss,bb) ==> Phonemes ==> Words ==> Sentences

So deep neural network with multiple hidden layers might be able to have the earlier layers learn these lower level simple features and then have the later deeper layers then put together the simpler things it's detected in order to detect more complex things like recognize specific words or even phrases or sentences.
Neural Researchers think that deep neural networks "think" like brains (simple ==> complex)

The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.

And what we see is that whereas the other layers are computing, what seems like relatively simple functions of the input such as where the edge is, by the time you get deep in the network you can actually do surprisingly complex things. Such as detect faces or detect words or phrases or sentences. Some people like to make an analogy between deep neural networks and the human brain, where we believe, or neuro-scientists believe, that the human brain also starts off detecting simple things like edges in what your eyes see then builds those up to detect more complex things like the faces that you see.

__B. Circuit Theory__

If you try to compute the same function with a shallow network, so if there aren't enough hidden layers, then you might require exponentially more hidden units to compute.

__C. Suggestion__

When starting on an application don't start directly by dozens of hidden layers. Try the simplest solutions (e.g. Logistic Regression), then try the shallow neural network and so on.

#### 5. Building blocks of deep neural networks (forward & backward Propagation)

![W4.4](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W4/W4.4.png?raw=true)

__cache__ here means __storing the value of Z in backward propagation__.

We use it to pass variables computed during forward propagation to the corresponding backward propagation step. It contains useful values for backward propagation to compute derivatives.

Sometime, we also cache weights (w) and biases (b) to make code more efficient.

![W4.5](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W4/W4.5.png?raw=true)

#### 6. Parameters vs Hyperparameters

What are hyperparameters?

- learning rate alpha
- '#' of iterations
- '#' of hidden layer L
- '#' of hidden units n[l]
- choice of activation function

We call all of these things above __hyper parameters__ because these things like alpha the learning rate the number of iterations number of hidden layers and so on these are all parameters that __control/determine the value of parameters(W and B)__


__Find the best value for hyperparameters__:

![W4.6](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W4/W4.6.png?raw=true)

It's very difficult to know in advance exactly what's the best value of the hyperparameters, so what often happen is you just have to try out many different values and go around this cycle your trial some value really try five hidden layers with this many number of hidden units implement that see if it works and then iterate. 

So the title of this slide is that apply deep learning is very __empirical process__ and empirical process is maybe a fancy way of saying you just have to try a lot of things and see what works.

Another effect I've seen is that deep learning today is applied to _unstructured data applications_ ranging from computer vision to speech recognition to natural language processing to a lot of _structured data applications_ such as maybe a online advertising or web search or product recommendations and so on.

__The best hyperparameters might change overtime__

Maybe you're working on online advertising as you make progress on the problem is quite possible there the best value for the learning rate a number of hidden units and so on might change so even if you tune your system to the best value of hyper parameters to daily as possible you find that the best value might change a year from now maybe because the computer infrastructure I'd be it you know CPUs or the type of GPU running on or something has changed and it is possible that the guidance won't to converge for some time and you just need to keep trying out different values and evaluate them on a hold on cross-validation set or something and pick the value that works for your problems.

On the next course we will see how to optimize hyperparameter in detail.

#### 7. What does deep learning have to do with the brain

Q: why people keep making the analogy between deep learning and the human brain?

A: It's been difficult to convey intuitions about what these equations are doing really gradient descent on a very complex function, __the analogy that is like the brain has become really an oversimplified explanation for what this is doing__, but the simplicity of this makes it seductive for people to just say it publicly, as well as, for media to report it, and certainly caught the popular imagination.

There is a loose analogy between logistic regression unit with a sigmoid activation function.

A single neuron appears to be much more complex than we are able to characterize with neuroscience, and while some of what is doing is a little bit like logistic regression, there's still a lot about what even a single neuron does that no human today understands. For example, exactly how neurons in the human brain learns, is still a very mysterious process. __It's completely unclear today whether the human brain uses an algorithm,does anything like back propagation or gradient descent or if there's some fundamentally different learning principle that the human brain uses__.

### II. Practice

[Building your Deep Neural Network - Step by Step](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step.ipynb)

[Deep Neural Network Application](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Deep%20Neural%20Network%20-%20Application.ipynb)

### III. Quiz

[Quiz_4](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/Quiz/W4%20Quiz.md)
