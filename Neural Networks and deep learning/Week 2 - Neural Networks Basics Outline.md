## Week 2 - Neural Networks Basics Outline

Learn to set up a machine learning problem with a neural network mindset. Learn to use vectorization to speed up your models.

### I. Logistic Regression as a Neural Network

__Note__: 

Logistic regression is used when the response variable is _categorical_ in nature.

Linear regression is used when your response variable is _continuous_.

#### 1. Binary classification

input: a pic of cats

output: 1(cat) vs 0 (non cat)

![W2.1](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.1.png?raw=true)

x- pixel intensity values;
y- {0, 1} non-cat, cat

__Notations__:

![W2.2](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.2.png?raw=true)

![W2.3](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.3.png?raw=true)

- `M is the number of training vectors`
- `Nx is the size of the input vector`
- `Ny is the size of the output vector`
- `X(1) is the first input vector`
- `Y(1) is the first output vector`
- `X = [x(1) x(2).. x(M)]`
- `Y = (y(1) y(2).. y(M))`

#### 2. Logistic regression

![W2.4](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.4.png?raw=true)

Equations:

Simple equation:	`y = wx + b`

- If x is a vector: `y = w.Tx + b`
- If we need y to be in between 0 and 1 (probability): `y = g(w.Tx + b)`

In binary classification `Y` has to be between `0` and `1`. Here we see Y as the probability of the chance that it equals to one given the input features X, Y=P(Y=1|X)

In the last equation `w` is a vector of `Nx` and `b` is a real number

#### 3. Logistic regression cost function

The superscript i in parentheses refers to data associated with the ith training example

![W2.5](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.5.png?raw=true)

First loss function would be the square root error:  `L(y',y) = 1/2 (y' - y)^2`

But we won't use this notation here for binary classification because it leads us to optimization problem which is non convex, means it contains local optimum points.

This is the function that we will use: `L(y',y) = - (y*log(y') + (1-y)*log(1-y'))`

To explain the last function lets see:
  
- if `y = 1` ==> `L(y',1) = -log(y')`  ==> we want `y'` to be the largest   ==> `y`' biggest value is 1
- if `y = 0` ==> `L(y',0) = -log(1-y')` ==> we want `1-y'` to be the largest ==> `y'` to be smaller as possible because it can only has 1 value.

Cost function will be: `J(w,b) = (1/m) * Sum(L(y'[i],y[i]))`

__Cost Function__: the cost of your parameters
So in training your logistic regression model, we're going to try to find parameters W and B that minimize the overall costs of machine J written at the bottom.

Q: __Difference between Loss Function and Cost Function__

S: __The Loss function__ computes the __error for a single training example__; __The cost function__ is the __avg of the loss functions of the entire training set__. 


#### 4. Gradient Descent

We want to predict `w` and `b` that minimize the cost function.

Our cost function is convex(no local optima).

First we initialize `w` and `b` to 0,0 or initialize them to a random value in the convex function and then try to improve the values the reach minimum value.

In Logistic regression people always use 0,0 instead of random.

The gradient decent algorithm repeats: `w = w - alpha * dw` where alpha is the learning rate and `dw` is the derivative of `w` (Change to `w`)

The derivative is also the slope of `w`

The derivative give us the direction to improve our parameters.

The actual equations we will implement:

- `w = w - alpha * d(J(w,b) / dw)` (how much the function slopes in the w direction)
- `b = b - alpha * d(J(w,b) / db)` (how much the function slopes in the d direction)

#### 5. Derivatives

- `f(a) = a^2`  ==> `d(f(a))/d(a) = 2a`
- `f(a) = a^3`  ==> `d(f(a))/d(a) = 3a^2`
- `f(a) = log(a)`  ==> `d(f(a))/d(a) = 1/a`

To conclude, Derivative is the slope and slope is different in different points in the function thats why the derivative is a function.

#### 6. Computation graph

Its a graph that organizes the computation from left to right.

![W2.6](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.6.png?raw=true)

#### 7. Logistic Regression Gradient Descent

In the video he discussed the derivatives of gradient decent example for one sample with two features `x1` and `x2`.

![W2.7](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.7.png?raw=true)

__Compute the derivative of Sigmoid__:

```
  s = sigmoid(x)
  ds = s * (1 - s) # derivative using calculus
```

![W2.8](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.8.png?raw=true)

To make an image of `(width,height,depth)` be a vector, use this:

```
  v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)  #reshapes the image.
```

Gradient descent converges faster after normalization of the input matrices.

#### 8. Gradient Descent on m Examples

Lets say we have these variables:

```
  X1					        Feature_1
  X2                  Feature_2
  W1                  Weight of the first feature.
  W2                  Weight of the second feature.
  B                   Logistic Regression parameter.
  M                   Number of training examples
  Y(i)				        Expected output of i
```

So we have:

![W2.9](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.9.png?raw=true)

Then from right to left we will calculate derivations compared to the result:

```
  d(a)  = d(l)/d(a) = -(y/a) + ((1-y)/(1-a))
  d(z)  = d(l)/d(z) = a - y
  d(W1) = X1 * d(z)
  d(W2) = X2 * d(z)
  d(B) = d(z)
```

From the above we can conclude the logistic regression pseudo code:

```
  	J = 0; dw1 = 0; dw2 =0; db = 0; # Devs
  	w1 = 0; w2 = 0; b=0; # Weights and bias
  	for i = 1 to m
  		
      # Forward propagation
  		z(i) = W1*x1(i) + W2*x2(i) + b
  		a(i) = Sigmoid(z(i))
  		J += (Y(i)*log(a(i)) + (1-Y(i))*log(1-a(i)))

  		# Backward propagation
  		dz(i) = a(i) - Y(i)
  		dw1 += dz(i) * x1(i)
  		dw2 += dz(i) * x2(i)
  		db  += dz(i)
  	
    J /= m
  	dw1/= m
  	dw2/= m
  	db/= m

  	# Gradient descent
  	w1 = w1 - alpa * dw1
  	w2 = w2 - alpa * dw2
  	b = b - alpa * db
```

The above code should run for some iterations to minimize error.

So there will be two inner loops to implement the logistic regression.

### II. Python and Vectorization

#### 1. Vectorization

__Vectorization__ is so important on deep learning to reduce loops.

Python vectorization. As observed in Rotating particles and Python efficiency, the speed of Python code can often be increased greatly by vectorizing mathematical expressions that are applied to NumPy arrays rather than using loops.

Machine Learning Explained: Vectorization and matrix operations. With vectorization these operations can be seen as matrix operations which are often more efficient than standard loops. Vectorized versions of algorithm are several orders of magnitudes faster and are easier to understand from a mathematical perspective.

![Loop vs Vectorized vs Scikit](file:///C:/Users/surface/Yinxiang%20Biji/Databases/Attachments/eb6925b7d80c4ed154b1320fb54f73e9.webp)

The vectorization can be done on CPU or GPU thought the SIMD operation. But its faster on GPU.

__Non-vectorized Computing__:

![W2.10](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.10.png?raw=true)

__Vectorized Computing__:

![W2.11](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.11.png?raw=true)

__Advanced Vectorization on Logistic Regression__:

![W2.12](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.12.png?raw=true)

As an input we have a matrix `X` and its `[Nx, m]` and a matrix `Y` and its `[Ny, m]`.

We will then compute at instance `[z1,z2...zm] = W' * X + [b,b,...b]`. This can be written in python as:

```python
    Z = np.dot(W.T,X) + b    # Vectorization, then broadcasting, Z shape is (1, m)
    
    A = np.sigmoid(Z)   # Vectorization, A shape is (1, m)
```

Vectorizing Logistic Regression's Gradient Output:

```python
   	dz = A - Y # Vectorization, dz shape is (1, m)
   	
    dw = np.dot(X, dz.T) / m # Vectorization, dw shape is (Nx, 1)
   	
    db = dz.sum() / m # Vectorization, dz shape is (1, 1)

    w := w - alpha*dw

    b := b - alpha*db
```

__Notice__:

However, if you need to implement multiple iterations of Gradient descend, you still need to use for loop.

#### 2. Notes on Python and NumPy

In NumPy, `np.sum(axis = 0)` sums the columns (sum along the rows) while `np.sum(axis = 1)` sums the rows (sum along the columns).

In NumPy, `np.reshape(1,4)` changes the shape of the matrix by broadcasting the values.

__Reshape__ is cheap in calculations so put it everywhere you're not sure about the calculations.

[Reshape in numpy](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Reshape%20in%20numpy.ipynb)

[Diff betw reshape & resize; flatten & ravel; squeeze](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Diff%20betw%20reshape%20%26%20resize%3B%20flatten%20%26%20ravel%3B%20squeeze.ipynb)

__Broadcasting__ works when you do a matrix operation with matrices that doesn't match for the operation, in this case NumPy automatically makes the shapes ready for the operation by broadcasting the values.

If you have an (m,n) matrix and you add(+) or subtract(-) or multiply(*) or divide(/) with a (1,n) matrix, then this will copy it m times into an (m,n) matrix. The same with if you use those operations with a (m , 1) matrix, then this will copy it n times into (m, n) matrix. And then apply the addition, subtraction, and multiplication of division element wise.

eg:

Q: Get the percentage of calories from carbs, proteins and fats for each of the four foods without using explicit for loops

![W2.13](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.13.png?raw=true)

S: ![W2.14](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.14.png?raw=true)

```
cal=A.sum( axis=0)
axis=0 means to sum vertically.

percentage = 100*A/(cal.reshape(1,4)) 
reshape to a 1x4 matrix
```

How does Python do [3x4 matrix]/[1x4 matrix]?

![W2.15](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.15.png?raw=true)

__General Principle__:

![W2.16](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W2/W2.16.png?raw=true)

Read more on [documentation of numpy](https://docs.scipy.org/doc/numpy/reference/index.html)

similar function in MATLAB/Octave: bsxfun

#### 3. A note on python/numpy vectors

__python/numpy__:

pros:

+ flexcible

cons:

+ easy to cause subtle bugs

__Don't use__:

a=np.random.randn(5)

"rank 1 array" - neither a row or a column vector

__Instead, use__:

a=np.random.randn(5,1)

a=np.random.randn(1,5)

1) make sure the vector is a mxn matrix
assert(a.shape == (m,n))

2) To reshape the vector into a mxn matrix:
a=a.reshape((m,n))

### III. Pieter Abbeel interview

so maybe, just like how deep learning for supervised learning was able to replace a lot of domain expertise, maybe we can have programs that are learned, that are reinforcement learning programs that do all this, instead of us designing the details.really try to see the connection from what you're working on to what impact they can really have, what change it can make rather than what's the math that happened to be in your work.places to get started:

* Andrej Karpathy's deep learning course 
* Berkeley's deep reinforcement learning course 
* Most important is to GET STARTED

Q: What are the things that deep reinforcement learning is already working really well at? 

S: I think, if you look at some deep reinforcement learning successes, it's very, very intriguing. For example, learning to play Atari games from pixels, processing this pixels which is just numbers that are being processed somehow and turned into joystick actions. Then, for example, some of the work we did at Berkeley was, we have a simulated robot inventing walking and the reward that it's given is as simple as the further you go north the better and the less hard you impact with the ground the better. And somehow it decides that walking slash running is the thing to invent whereas, nobody showed it, what walking is or running is. Or robot playing with children's stories and learn to kind of put them together, put a block into matching opening, and so forth.It's nice it learns from scratch for each one of these tasks but would be even nicer if it could reuse things it's learned in the past; to learn even more quickly for the next task. And that's something that's still on the frontier and not yet possible. It always starts from scratch, essentially.


### IV. Practice

[Python Basics with numpy](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Python%20Basics%20With%20Numpy.ipynb)

[logistic Regression with a Neural Network mindset](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Build%20your%20first%20image%20recognition%20algorithm.ipynb)

### V. Quiz

[Quiz_2](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/Quiz/W2%20Quiz.md)