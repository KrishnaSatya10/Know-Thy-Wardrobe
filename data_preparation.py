#Importing libraries
import torch
from config import LinearClassifierConfig
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class DataPreparation:
    """The base class of data."""
    def __init__(self, root="data", batch_size=64, num_classes = 10, image_size = (32,32), transform=True, target_transform=True):
        self.config = LinearClassifierConfig(root = root, num_classes = num_classes, batch_size = batch_size, image_size = image_size)
        if transform:
          self.transform = transforms.Compose([transforms.Resize(self.config.image_size),transforms.ToTensor()])
        if target_transform:
          self.target_transform = transforms.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
  

    def load_training_data(self):
      """
        The FashionMNIST features are in PIL Image format, and the labels are integers. 
        For training, we need the features as normalized tensors, and the labels as one-hot encoded tensors. 
        To make these transformations, we use ToTensor and Lambda.

        Lambda Transforms
        Lambda transforms apply any user-defined lambda function. Here, we define a function to turn the integer into a one-hot encoded tensor. 
        It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.
      """
      self.training_data = datasets.FashionMNIST(
          root=self.config.root,
          train=True,
          download=True,
          transform=self.transform,
          target_transform=self.target_transform
          )


    def load_test_data(self):
      self.test_data = datasets.FashionMNIST(
          root=self.config.root,
          train=False,
          download=True,
          transform=self.transform,
          target_transform=self.target_transform
          )
        

    def train_dataloader(self):
      # Preparing your data for training with DataLoaders
      """
      The ``Dataset`` retrieves our dataset's features and labels one sample at a time. 
      While training a model, we typically want to pass samples in "minibatches", reshuffle the data at every epoch to reduce model overfitting, 
      and use Python's ``multiprocessing`` to speed up data retrieval.
      # ``DataLoader`` is an iterable that abstracts this complexity for us in an easy API.
      """

      # Iterate through the DataLoader
      """
      We have loaded that dataset into the ``DataLoader`` and can iterate through the dataset as needed.
      Each iteration below returns a batch of ``train_features`` and ``train_labels`` (containing ``batch_size=64`` features and labels respectively).
      Because we specified ``shuffle=True``, after we iterate over all batches the data is shuffled (for finer-grained control over
      the data loading order, take a look at `Samplers <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_).
      """
      self.train_dataloader = DataLoader(self.training_data, batch_size = self.config.batch_size, shuffle=True)
      
    def test_dataloader(self):
      self.test_dataloader = DataLoader(self.test_data, batch_size = self.config.batch_size, shuffle=True)

    def visualize_data(self,labels_map):
      """
      Display image and label.
      #input_tensor.squeeze() -- Returns a tensor with all specified dimensions of input of size 1 removed.
      #For example, if input is of shape: (A×1×B×C×1×D) then the input.squeeze() will be of shape: (A×B×C×D).
      """

      train_features, train_labels = next(iter(self.train_dataloader))
      print(f"Feature batch shape: {train_features.size()}")
      print(f"Labels batch shape: {train_labels.size()}")
      img = train_features[0].squeeze()
      label = (train_labels[0] == 1).nonzero(as_tuple = False).item()
      print(label)

      plt.title(labels_map[label])
      plt.imshow(img, cmap="gray")
      plt.show()
      print(f"Label: {label}")

      