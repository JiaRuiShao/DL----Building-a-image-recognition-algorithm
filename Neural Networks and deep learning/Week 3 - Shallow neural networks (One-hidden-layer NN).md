## Week 3 - Shallow neural networks (One-hidden-layer NN)

Learn to build a neural network with one hidden layer, using forward propagation and backpropagation.

### I. Shallow Neural Network

#### 1. Neural Network Overview

![W3.1](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.1.png?raw=true)

First, we'll inputs the features, x, together with some parameters w and b, and this will allow you to compute z one. So, new notation that we'll introduce is that we'll use superscript square bracket one to refer to quantities associated with this stack of nodes, it's called a layer. Then later, we'll use superscript square bracket two to refer to quantities associated with that node. That's called another layer of the neural network. 

The superscript square brackets, like we have here, are not to be confused with the superscript round brackets which we use to refer to individual training examples.

Whereas x superscript round bracket I refer to the ith training example, superscript square bracket one and two refer to these different layers; layer one and layer two in this neural network.

Q: What neural network looks like?

S: It's basically taking logistic regression and repeating it twice.

#### 2. Notation

![W3.2](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.2.png?raw=true)

Input layer
Hidden layer
Output layer

In a neural network that you train with supervised learning, the training set contains values of the inputs x as well as the target outputs y.So the term __hidden layer refers to the fact that in the training set, the true values for these nodes in the middle are not observed__. That is, you don't see what they should be in the training set. You see what the inputs are and what the output should be. But the things in the hidden layer are not seen in the training set. 

Whereas previously, we were using the vector X to denote the input features and alternative notation for the values of the input features will be _A superscript square bracket 0_. And the term __A__ also stands for __activations__, and it __refers to the values that different layers of the neural network are passing on to the subsequent layers__. So the input layer passes on the value x to the hidden layer, so we're going to call that activations of the input layer _A super script 0_. The next layer, the hidden layer, will in turn generate some set of activations, which I'm going to write as _A superscript square bracket 1_. So in particular, this first unit or this first node, we generate a value _A superscript square bracket 1_. This second node we generate a value. Now we have a subscript 2 and so on. And so, A superscript square bracket 1, this is a four dimensional vector you want in Python because the 4x1 matrix, or a 4 column vector, which looks like this. And it's four dimensional, because in this case we have four nodes, or four units, or four hidden units in this hidden layer. And then finally, the open layer regenerates some value A2, which is just a real number. And so y hat is going to take on the value of A2. So this is analogous to how in logistic regression we have y hat equals a and in logistic regression which we only had that one output layer, so we don't use the superscript square brackets. But with our neural network, we now going to use the superscript square bracket to explicitly indicate which layer it came from.

This network that you've seen here is called a __two layer neural network__.And the reason is that when we count layers in neural networks, we don't count the input layer. So the hidden layer is layer one and the output layer is layer two. In our notational convention, we're calling the input layer layer zero, so technically maybe there are three layers in this neural network. But in conventional usage, if you read research papers and elsewhere in the course, you see people refer to this particular neural network as a two layer neural network, because we don't count the input layer as an official layer.

#### 3. Computing a Neural Network Output

![W3.3](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.3.png?raw=true)

stack w vector together (vertically), we end up with a matrix:

![W3.4](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.4.png?raw=true)

![W3.5](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.5.png?raw=true)

Summarize:

![W3.6](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.6.png?raw=true)

![W3.7](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.7.png?raw=true)

#### 4. Vectorizing across multiple examples

![W3.8](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.8.png?raw=true)

![W3.9](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.9.png?raw=true)

The value at the top left most corner of the mean corresponds to the activation of the first heading unit on the first training example. One value down corresponds to the activation in the second hidden unit on the first training example, then the third heading unit on the first training sample and so on. So as you scan down this is your indexing to the hidden units number.

Whereas if you move horizontally, then you're going from the first hidden unit. And the first training example to now the first hidden unit and the second training sample, the third training example. And so on until this node here corresponds to the activation of the first hidden unit on the final train example and the nth training example.

#### 5. Explanation for Vectorized Implementation

To simplify the justification, we assume b[1]=0

![W3.10](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.10.png?raw=true)

![W3.11](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.11.png?raw=true)

#### 6. Non-linear Activation Function (sigmoid function is only one of the many activation functions)

![W3.12](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.12.png?raw=true)

__A. Sigmoid Function__

Sigmoid activation function range is [0,1] 

`A = 1 / (1 + np.exp(-z)) # Where z is the input matrix`

![Sigmoid function](https://cdn-images-1.medium.com/max/1600/1*Xu7B5y9gp0iL5ooBj7LtWw.png)

- never use this, except that you are doing binary classification
- tangent function is superior

Sigmoid outputs a value between 0 and 1 which makes it a very good choice for binary classification. You can classify as 0 if the output is less than 0.5 and classify as 1 if the output is more than 0.5. It can be done with tanh as well but it is less convenient as the output is between -1 and 1.

__B. Tangent function/hyperbolic tangent function__

Tanh activation function range is [-1,1]   (Shifted version of sigmoid function)

![Tangent](https://1.bp.blogspot.com/-yFDcusHo-BM/WlIwsZ96xqI/AAAAAAAAAoc/M3YjIaNt_poi_r1Kkkhe4nxJWEakYXxkACPcBGAYYCw/s1600/Hyperbolic-tangent-function.JPG)

In NumPy we can implement Tanh using one of these methods:

```python
    `A = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z)) # Where z is the input matrix`
```

OR

```python
    A = np.tanh(z)   # Where z is the input matrix
```

It turns out that the tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer.

__Cons for sigma and tangent activation function__:

- if z is either very large or very small, then the gradient or the derivative or the slope of this function becomes very small. So if z is very large or z is very small, the slope of the function ends up being close to 0, which can slow down the gradient descent

Alternative for these two activation function:

__C. Rectify linear unit/ReLU function__

![ReLu and Leaky Relu](https://www.codeproject.com/KB/AI/1220276/ReLU.png)

- a=max(0, z)
- the most commonly used activation function
- if you're not sure what else to use, use this one
- don't suffer from vanishing gradients problem
- fast to compute

For larger values of z, tan h and sigmoid will have slope = 0, thus they will take large amount of time in reaching optimal solution in gradient descent whereas how much large my z will be i will have a constant value for slope of z in ReLU which help my gradient descent to reach optimum solution faster than tan h or sigmoid.

__D. Leaky ReLU Function__

```
g(z)=0.01z, z<0
         z, z>=0
```

![Activation Functions](https://cdn-images-1.medium.com/max/1600/1*DRKBmIlr7JowhSbqL6wngg.png)

#### 7. Why using non-linear activation function?

if using linear activation function, then:

```
	a[1]=z[1]=w[1]x+b[1]
	a[2]=z[2]=w[2]a[1]+b[2]=w[2]*(w[1]x+b[1])+b[2]
		=(w[2]w[1])x + (w[2]b[1]+b[2]) = w'x+b'
```

If you were to use __linear activation functions__ or we can also call them __identity activation functions__, then the neural network is just outputting a linear function of the input.

It turns out that if you use a linear activation function or alternatively, if you don't have an activation function, then no matter how many layers your neural network has, all it's doing is just computing a linear activation function. So you might as well not have any hidden layers.

__A linear hidden layer is more or less useless because the composition of two linear functions is itself a linear function__.

__Exception__:

There is just one situation where you might use a __linear activation function__. g(x) = z. And that's if you are __doing machine learning on the regression problem__. 

The one place you might use a linear activation functions usually in the __output layer__ (use Relu/Tangent Function in hidden units). But other than that, using a linear activation function in the hidden layer except for some very special circumstances relating to compression that we're going to talk about using the linear activation function is extremely rare.

#### 8. Derivatives of activation functions

__A. Sigmoid Function__

```
	g(z)=1/(1+e^-z)
	g'(z)=g(z)*(1-g(z))=a(1-a)
```

__B. Tangent Function__

```
	g(z)=tanh(z)=(e^z - e^-z)/(e^z + e^-z)
	g'(z)=1-g(z)^2
```

__C. ReLU Function__

```
	g(z)=max(0,z)
	g'(z)= 0, z < 0 
           z, z >=0
```

[More about Activation Functions](https://towardsdatascience.com/secret-sauce-behind-the-beauty-of-deep-learning-beginners-guide-to-activation-functions-a8e23a57d046)

#### 9. Computing gradients in two-layer neural network (Backward propagation)

[BackPropagation_1](http://colah.github.io/posts/2015-08-Backprop/)

[BackPropagation_2](https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/)

__chain rule of calculus__

![W3.15](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.15.png?raw=true)
(For supervised learning, we son't have to optimize input X)

![W3.16](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.16.png?raw=true)

Make sure dim of derivatives match up!!

![W3.17](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.17.png?raw=true)

__Summary of gradient descent__

![W3.18](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W3/W3.18.png?raw=true)

we use (keepdims = True) to make sure that A.shape is (4,1) and not (4, ). It makes our code more rigorous.

### II. Ian Goodfellow Interview

I think one thing that I got from your courses at Stanford is that __linear algebra and probability__ are very important, that people get excited about the machine learning algorithms, but if you want to be a really excellent practitioner, you've got to master the basic math that underlies the whole approach in the first place. So we make sure to give a very focused presentation of the math basics at the start of the book. That way, you don't need to go ahead and learn all that linear algebra, that you can get a very quick crash course in the pieces of linear algebra that are the most useful for deep learning.

And now, we're at a point where there are so many different paths open that someone who wants to get involved in AI, maybe the hardest problem they face is choosing which path they want to go down. Do you want to make reinforcement learning work as well as supervised learning works? Do you want to make unsupervised learning work as well as supervised learning works? Do you want to make sure that machine learning algorithms are fair and don't reflect biases that we'd prefer to avoid? Do you want to make sure that the societal issues surrounding AI work out well, that we're able to make sure that AI benefits everyone rather than causing social upheaval and trouble with loss of jobs? I think right now, there's just really an amazing amount of different things that can be done, both to prevent downsides from AI but also to make sure that we leverage all of the upsides that it offers us.

Q: what advice would you have for someone like that? 

S: I think a lot of people that want to get into AI start thinking that they absolutely need to get a Ph.D. or some other kind of credential like that. I don't think that's actually a requirement anymore. One way that you could get a lot of attention is to write good code and put it on GitHub. If you have an interesting project that solves a problem that someone working at the top level wanted to solve, once they find your __GitHub repository__, they'll come find you and ask you to come work there. A lot of the people that I've hired or recruited at OpenAI last year or at Google this year, I first became interested in working with them because of something that I saw that they released in an open-source forum on the Internet. __Writing papers and putting them on Archive__ can also be good.

__I think if you learned by reading the book, it's really important to also work on a project at the same time, to either choose some way of applying machine learning to an area that you are already interested in__. Like if you're a field biologist and you want to get into deep learning, maybe you could use it to identify birds, or if you don't have an idea for how you'd like to use machine learning in your own life, you could pick something like making a Street View house numbers classifier, where all the data sets are set up to make it very straightforward for you. And that way, you get to exercise all of the basic skills while you read the book or while you watch Coursera videos that explain the concepts to you.

Q: So over the last couple of years, I've also seen you do one more work on adversarial examples. Tell us a bit about that. 

S: Yeah. I think adversarial examples are the beginning of a new field that I call machine learning security. 

In the past, we've seen computer security issues where attackers could fool a computer into running the wrong code. That's called __application-level security__. 

And there's been attacks where people can fool a computer into believing that messages on a network come from somebody that is not actually who they say they are. That's called __network-level security__.

Now, we're starting to see that you can also fool machine-learning algorithms into doing things they shouldn't, even if the program running the machine-learning algorithm is running the correct code, even if the program running the machine-learning algorithm knows who all the messages on the network really came from. And I think, _it's important to build security into a new technology near the start of its development_. We found that it's very hard to build a working system first and then add security later. So I am really excited about the idea that if we dive in and start anticipating security problems with machine learning now, we can make sure that these algorithms are secure from the start instead of trying to patch it in retroactively years later.

### III. Practice 

[Planar data classification with one hidden layer](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/projects/Planar%20data%20classification%20with%20one%20hidden%20layer.ipynb)

### IV. Quiz

[Quiz_3](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/Quiz/W3%20Quiz.md)