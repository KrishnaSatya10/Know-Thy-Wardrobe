#Importing libraries
import torch

class LinearClassifierConfig:
  def __init__(self, root, num_classes, batch_size, image_size, optimizer = torch.optim.SGD, loss_function = torch.nn.CrossEntropyLoss):
        self.root = root
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.optimizer = optimizer
        self.loss_function = loss_function
    
  def do_something(self):
    raise NotImplementedError
   