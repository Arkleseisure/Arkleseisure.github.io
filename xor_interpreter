import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
from interp_lib import *  # Import custom library containing MLP, device, find_trigger_input, etc.

# Record the start time of the script
start_time = time.time()
print(f'Start time: {time.ctime(start_time)}')
    
# Hyperparameters
batch_size = 1        # Number of samples per batch
learning_rate = 0.001 # Learning rate for the optimizer
epochs = 2000         # Number of training epochs
width = 4             # Width of the neural network (number of neurons per layer)
depth = 2             # Depth of the neural network (number of hidden layers)

# Define a custom dataset for the XOR problem
class XORDataset(Dataset):
    def __init__(self):
        # Define the XOR input data and corresponding labels
        self.data = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ], dtype=torch.float32)
        
        self.labels = torch.tensor([
            [0],
            [1],
            [1],
            [0]
        ], dtype=torch.float32)

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, index):
        # Return the data and label at the specified index
        return self.data[index], self.labels[index]

# Instantiate the XOR dataset for training and testing
train_dataset = XORDataset()
test_dataset = XORDataset()

# Create data loaders for the training and test datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the loss function (Mean Squared Error Loss)
criterion = nn.MSELoss()
    

def train_model():
    # Instantiate the neural network model and move it to the specified device (CPU or GPU)
    model = MLP(2, 1, width, depth, 'relu').to(device)
    # Define the optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Variable to accumulate the loss
        
        # Iterate over the training data
        for images, labels in train_loader:
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()       # Zero the parameter gradients
            outputs, intermediates = model(images)  # Forward pass
            loss = criterion(outputs, labels)       # Compute loss
            loss.backward()                         # Backward pass
            optimizer.step()                        # Optimize the model parameters
            running_loss += loss.item()             # Accumulate loss

        # Calculate average training loss for the epoch
        train_loss = running_loss / len(train_loader)

        # Print training progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Loss: {train_loss:.4f}, ')
            print(f'Total time: {time.time() - start_time}')
            print(f'Current time: {time.ctime(time.time())}')

    return model, train_loss  # Return the trained model


# Gets the inputs which activate the various features of the model
def get_trigger_inputs_outputs(model):
    # List to store inputs that maximize each feature's activation
    trigger_inputs = []
    for i in range(depth):
        trigger_inputs.append([])
        for j in range(width):
            # Find an input that maximizes the activation of the jth feature of the ith layer
            trigger_inputs[i].append(find_trigger_input(i, j, model))

    trigger_outputs = [[model(torch.tensor(trigger_inputs[i][j], dtype=torch.float32).to(device))[1][i][j] for j in range(width)] for i in range(depth)]
    for i in range(depth):
        print('Layer:', i + 1)
        for j in range(width):
            print('Feature', j + 1)
            print('Trigger input:', trigger_inputs[i][j].tolist())
            print('Value with trigger input:', trigger_outputs[i][j].item())
    return trigger_inputs, trigger_outputs
    

def get_feature_responses(model):
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]  # XOR input data
    feature_responses = []  # List to store feature activations for each input
    outputs = []            # List to store model outputs for each input
    
    # Evaluate the model on each input data point
    for i in range(len(data)):
        # Convert input data to a tensor and move it to the device
        output, features = model(torch.tensor(data[i], dtype=torch.float32).to(device))
        outputs.append(output.item())               # Store the output
        feature_responses.append([features[j].tolist() for j in range(len(features))]) # Store the feature activations
        print('Input:', data[i])
        print('Output:', output.item())
        print('Features:', feature_responses[-1])
        
    return data, outputs, feature_responses


def get_model_params(model):
    # Collect model parameters for interpretation
    params = []
    for name, param in model.named_parameters():
        print(name)
        print(param)
        params.append(param)
        
    return params

def build_content_prompt(trigger_inputs, trigger_outputs, data, outputs, feature_responses, params, train_loss):
    # Build a content prompt for ChatGPT to interpret the model's features
    content_prompt = f'The neural network is an mlp which has {depth} hidden layers and width {width}. It has final loss {train_loss} (loss function {type(criterion)}) and relu activations.\n'
    for i in range(depth):
        content_prompt += 'This is for layer {i}:\n'
        for j in range(width):
            content_prompt += f' These are the weights for feature {j + 1}: \n'
            content_prompt += f' {params[2 * i][j].tolist()}\n'   # Weights of the first layer
            content_prompt += f' This is the bias for feature {j + 1}: \n'
            content_prompt += f' {params[2 * i + 1][j].tolist()}\n'   # Biases of the first layer
            content_prompt += f" This is the feature's response to some input data:\n"
            for k in range(len(data)):
                content_prompt += f' Input: {data[k]} \n Feature response: {feature_responses[j][i][k]}\n'
            content_prompt += f' This is an input designed to maximise the output of the feature:\n'
            content_prompt += f' Input: {trigger_inputs[i][j].tolist()}, Output activation of feature: {trigger_outputs[i][j].item()}\n'
            content_prompt += f' Please give an interpretation of what you think this feature means, and use reasoning based on the data given.\n'
    content_prompt += f' Once you have found the values of these features, use the following data to summarise how you think the rest of the network works:\n'
    content_prompt += f' Final layer weights: {params[-2].tolist()}\n'  # Weights of the final layer
    content_prompt += f' Final layer biases: {params[-1].tolist()}\n'   # Biases of the final layer
    content_prompt += f' Outputs:\n'
    for i in range(len(data)):
        content_prompt += f'Input: {data[i]}, Output: {outputs[i]}\n'
    content_prompt += f' Finally summarise overall how the network works.'
    
    return content_prompt

def evaluate_network(model, train_loss):
    # Gets the inputs which trigger features in the network, and the values they output when triggered
    trigger_inputs, trigger_outputs = get_trigger_inputs_outputs(model)

    # Gets the feature responses to the data points
    data, outputs, feature_responses = get_feature_responses(model)
    
    # returns a list of the parameter of the model
    params = get_model_params(model)

    # Builds a prompt to get chatgpt to interpret the functioning of the network
    content_prompt = build_content_prompt(trigger_inputs, trigger_outputs, data, outputs, feature_responses, params, train_loss)
    
    # Print the content prompt for debugging purposes
    print('\n\n\n\n\n' + content_prompt)

    # Send the prompt to ChatGPT and get an interpretation of the features
    response = get_chatgpt_response(content_prompt)

    # Print the response from ChatGPT
    print(response)

def main():
    # Train the model on the XOR dataset
    model, train_loss = train_model()
    # Evaluate the trained model and interpret its features
    evaluate_network(model, train_loss)

# Execute the main function when the script is run
if __name__ == '__main__':
    main()
