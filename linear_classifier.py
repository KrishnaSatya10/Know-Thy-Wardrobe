# Importing libraries
import torch
from torch import nn
from config import LinearClassifierConfig


class LinearClassifier(nn.Module):
    def __init__(self, root, batch_size, num_classes, image_size):
        super().__init__()
        """
        We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values
        The minibatch dimension (at dim=0) is maintained. So a 64 * 28 * 28 becomes 64 * 784

        Skipping Lazy Linear and using Linear Instead to Illustrate that this 784 size will now go as input to a simple linear layer which is the input layer, and the output is a 10 node layer, one corresponding to each class. 
        Specifying this is automatically creating the parameters for us, i.e. the weights and biases with the required dimension. 

        Therefore in Pytorch we can just specify the layer-to-layer architecture and Pytorch will automatically define the weights and bias variables that fit with these inputs. And since it also handles the gradient computation during backprop using autograd it is able to neatly abstract away a lot of logic from us. 
        """
        self.config = LinearClassifierConfig(
            root=root,
            num_classes=num_classes,
            batch_size=batch_size,
            image_size=image_size,
        )

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.config.image_size[0] * self.config.image_size[1],
                self.config.num_classes,
            ),
        )
        print(f"Model structure: {self.model}")

        for (
            name,
            param,
        ) in (
            self.model.named_parameters()
        ):  # nn.Module.parameters() returns an iterator over module parameters.
            print(
                f"Parameter name: {name} | Size: {param.size()} | Values : {param[:2]} \n"
            )

    def forward(self, train_features_batch):
        """
        This function passes the input features batch through the network and returns the output, in this case it is the logits, i.e. the unnormalized outputs of the network. It doesn't have softmax applied to it yet.
        """
        logits = self.model(train_features_batch)
        return logits

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate)

    def compute_loss(self, logits, train_labels):
        loss_function = nn.CrossEntropyLoss()
        """
      This criterion computes the cross entropy loss between input logits and target.
      The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).
      It is useful when training a classification problem with C classes.
      """
        return loss_function(logits, train_labels)
