In this repo I am to implement Deep-Learning based image classification on the FashionMNIST dataset which contains imgaes of clothing items in gray scale (28 x 28). I implement the classigication using:

- Linear Classification based Neural Network (no hidden layers)
- Deep Learning based Neural Network
- Convolutional Neural Network

Further, for each of these 3 variants I seek to implement 2 modes:

- From scratch
- using built-in Pytorch libraries for concise implementation

I seek to design my implementation to adhere to Pytorch-ic guideline which uses object oriented approach to building neural networks, by representing the components of deep learning as objects and methods that define interactions between objects. This paradigm has been adopted after studying multiple Pytorch based NN codes and larger context of open-source frameworks used for deep=learning -- For Pytorch base my knowledge on Official PyTorch Documentation as well as the book Dive Into Deep Learning, which is an excellent resource for those who wish to learn Pytorch for Deep Learning.

That said, my code revolves using Methods, Constructors and Utility functions (as evident for the networks built from scratch) primarily belonging to 3 major Classes for developing the end-to-end model:

(i) Module - contains models, losses, and optimization methods;
(ii)DataModule - provides data loaders for training and validation;
(iii) Trainer - both classes are combined using the Trainer class, which allows us to train models on a variety of hardware platforms.

Anyways, the read-me will be periodically updated as I make progress. Wish me luck!
