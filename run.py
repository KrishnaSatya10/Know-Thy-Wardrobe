import argparse
import torch
from linear_classifier import LinearClassifier
from data_preparation import DataPreparation
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Your script description here.")
    parser.add_argument("--model_name", type=str, required=True, help="Specify the model name.")
    parser.add_argument("--device", type=str, default=None, help="Specify the device (default: cpu).")
    parser.add_argument("--root", type=str, default="data", help="Specify the root data folder.")
    parser.add_argument("--max_epochs", type=int, default=2, help="Specify the max number of epochs.")

    args = parser.parse_args()

    #Variables
    model_name = args.model_name
    device = (("cuda" if torch.cuda.is_available() else "cpu") if not args.device else args.device)
    root = args.root
    max_epochs = args.max_epochs
    batch_size = 64
    num_classes = 10
    image_size = (28,28)
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    
    #Data
    data_preparation = DataPreparation(root, batch_size, num_classes, image_size)        
    data_preparation.visualize_data(labels_map)
    
    #Model
    if model_name == "Linear Classifier":
        model = LinearClassifier(root, batch_size, num_classes, image_size).to(device)
        print(f"Running {model_name} on {device}")
    else:
        print("Model to be implemented")

    #Training
    trainer = Trainer(max_epochs, device)
    trainer.train_model(data_preparation, model)



if __name__ == "__main__":
    main()


#Command line should be like this
#python run.py --model_name model_a --device cuda