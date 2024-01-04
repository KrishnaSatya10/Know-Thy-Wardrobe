README IS WIP - AS I MAKE PROGRESS IN MY UNDERSTANDING OF PYTORCH AND DL, I WILL KEEP UPDATING THIS IN THE PROJECT

In this repo I seek to implement Deep-Learning based image classification on the FashionMNIST dataset which contains imgaes of clothing items in gray scale (28 x 28). I implement the classification using:

- Linear Classification based Neural Network (no hidden layers)
- Deep Learning based Neural Network
- Convolutional Neural Network

I seek to design my implementation to adhere to Pytorch-ic guideline which uses object oriented approach to building neural networks, by representing the components of deep learning as objects and methods that define interactions between objects. This paradigm has been adopted after studying multiple Pytorch based NN codes and larger context of open-source frameworks used for deep=learning (mxnet, jax, tensorflow) -- For Pytorch base my knowledge on Official PyTorch Documentation as well as the book Dive Into Deep Learning, which is an excellent resource for those who wish to learn Pytorch for Deep Learning.

That said, my code revolves using Methods, Constructors and Utility functions (as evident for the networks built from scratch) primarily belonging to 3 major Classes for developing the end-to-end model:

(i) Module (LinearClassifier, DeepClassifier, CNNClassifier) - contains models, losses, and optimization methods;

(ii) DataPreparation - provides data loaders for training and validation;

(iii) Trainer - both classes are combined using the Trainer class, which allows us to train models on a variety of hardware platforms.

Anyways, the read-me will be periodically updated as I make progress. Wish me luck!

I will maintain separate branches for every model I create -- the idea is to mimic the "features" we see in Jira and coding up a feature -- master branch needs to be clean while I try out ideas in the individual feature branches. Once I push working code to my satisfaction I plan to merge it with Master.

GITHUB ACTIONS:
I plan to use github actions to apply some checks before pull requests are merged to master. Specifically:

- Am I able to run the model successfully to the end
- Are all the black formatting checks passing?
- Are there any linter errors?

I need to learn how to implement github actions for this. Okay! Let's learn something new!!

OKAY, THE FOLDER STRUCTURE I HAVE IN MIND:

- config.py
- trainer.py
- data_preparation.py
- linear_classifier.py
- deep_classifier.py
- cnn_lassifier.py
- run.py
- Notes.txt
