## Week 1 - Introduction to deep learning

This Course is first of these five courses which teach you __the most important building blocks of deep learning__. 

By the end of this first course, you know how to build and get to work a deep neural network.

#### 1. What is a (Neural Network) NN?

Single neuron == linear regression

__Simple NN__:

![W1.1](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.1.jpg?raw=true)

__Relu function__ (Rectified Linear Unit)

![W1.2](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.2.jpg?raw=true)

RELU function is the most popular activation function right now that makes deep NNs train faster.

Big neural network is consisted of many small neurons stacking together. By stacking together a few of the single neurons or the simple predictors we have from the previous slide, we now have a slightly larger neural network. 

__Hidden layers__

Hidden layers predicts connection between inputs automatically, thats what deep learning is good at.

![W1.3](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.3.jpg?raw=true)

Here we have a neural network with four inputs. So the input features might be the size, number of bedrooms, the zip code or postal code, and the wealth of the neighborhood. And so given these input features, the job of the neural network will be to predict the price y. 

Deep NN consists of more hidden layers (Deeper layers)

![W1.4](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.4.png?raw=true)

Notice that each of these circle in the middle, these are called __hidden units__ in the neural network, that __each of them takes its inputs all four input features__. Rather than saying these first nodes represent family size and family size depends only on the features X1 and X2. Instead, we're going to say neural networks decide whatever it wants this known to be. And we'll give it all four of the features to complete whatever it wants.

And the remarkable thing about neural networks is that, __given enough training examples with both x and y, neural networks are remarkably good at figuring out functions that accurately map from x to y__.

#### 2. Supervised learning with neural networks

Input|Output|Application|Used types of neural networks
-----|------|-----------|-----------------------------
Home features|price|Real Estate|Standard NN
Ad, user info|Click on ad? (0/1)|Online Advertising|Standard NN
Image|Object (1,...1000)|Photo tagging|CNN
Audio|Text Transcript|Speech recognition|RNN
English|Chinese|Machine translation|RNN
Image, Radar info|Position of other cars on the road|Autonomous driving|Custom/Hybrid

__Neural Network Examples__

- **Standard NN**

- **Convolutional Neural Network (CNN)** (often used for image data)

- **Recurrent Neural Network (RNN)** (one-dimensional temporal sequence data; It is applicable when the input/output is a sequence)

![W1.5](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.5.jpg?raw=true)

__Supervised Learning__

- Structured Data (each of the features are well-defined)
- Unstructured Data(audio, image, text)

![W1.6](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.6.jpg?raw=true)

Historically, it has been much harder for computers to make sense of unstructured data compared to structured data.

Thanks to deep learning, thanks to neural networks, computers are now much better at interpreting unstructured data as well compared to just a few years ago.

For the purposes of explaining the algorithms, we will draw a little bit more on examples that use unstructured data. But as you think through applications of neural networks within your own team I hope you find both uses for them in both structured and unstructured data.

#### 3. Why is deep learning taking off?

![W1.7](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/images/W1/W1.7.jpg?raw=true)

- For small data NN can perform as Linear regression or SVM (Support vector machine)
- For big data a small NN is better that SVM
- For big data a big NN is better that a medium NN is better that small NN.

Deep learning is taking off for 3 reasons:

__1. Scale__
     
    scale of the (training set) data and size of the neural network

__2. Computation__
     
     fasten computation
     iterate much fasten
     try more ideas

     - GPUs.
     - Powerful CPUs.
     - Distributed computing.
     - ASICs

__3. Algorithm:__
     
    Creative algorithms has appeared that changed the way NN works
        
    For example using RELU function is so much better than using SIGMOID function in training a NN because it helps with the vanishing gradient problem.

### Quiz

[Quiz_1](https://github.com/JiaRuiShao/Deep-Learning/blob/DL/Neural%20Networks%20and%20deep%20learning/Quiz/W1%20Quiz.md)