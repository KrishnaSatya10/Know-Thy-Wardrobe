class Trainer:
    """
    The base class for training models with data.
    """

    def __init__(self, max_epochs, device):
        self.max_epochs = max_epochs
        self.device = device

    def prepare_data(self, data_preparation):
        data_preparation.load_training_data()
        data_preparation.load_test_data()

    def visualize_data(self, labels_map, data_preparation):
        data_preparation.visualize_data(labels_map)

    def prepare_model(self, model):
        self.model = model
        print("Printing the model template", model)

    def train_model(self, data_preparation, model):
        self.prepare_data(data_preparation)
        self.prepare_model(model)
        self.optimizer = self.model.configure_optimizers()

        self.model.train()  # Setting the model in train mode.

        for epoch in range(self.max_epochs):
            for batch, (train_features, train_labels) in enumerate(
                data_preparation.train_dataloader()
            ):
                # Forward step
                train_features, train_labels = train_features.to(
                    self.device
                ), train_labels.to(self.device)
                logits = self.model(train_features)
                loss_value = self.model.compute_loss(logits, train_labels)

                # Backpropagation
                self.optimizer.zero_grad()  # Setting the gradient to 0 for the batch
                loss_value.backward()
                self.optimizer.step()

                # Every 100 steps print the loss value:
                if batch % 100 == 0:
                    print(
                        f"Epoch: {epoch} | Batch Number: {batch} | Loss: {loss_value}"
                    )
