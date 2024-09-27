# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time
import openai
import wandb
import math
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests
import base64
import os  # For environment variables

# Initialize OpenAI API
# It's recommended to store your API key securely, e.g., using environment variables
# Get an openai api key from https://openai.com/api/ and then set an environment variable called "OPENAI_API_KEY" and set it to the key
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device being used: {device}')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
dataset = 'MNIST'
activation = 'relu'

# Set the parameters
epoch_nbr = 5
width = 6
depth = 1 # number of hidden layers
input_size = 28*28
output_size = 10

# Are we using weights and biases to track the experiment? This requires a (free) wandb account.
use_wandb = False

# Base prompt for GPT evaluation
gpt_prompt_base = (
    'The neural network we are evaluating has the following properties:\n'
    f'It is an MLP neural network with {depth} hidden layers and width {width}.\n'
    f'It is trained on the MNIST dataset and uses {activation} activations.\n'
)

# Initialize Weights & Biases (wandb) for experiment tracking
if use_wandb:
    wandb.init(
        project='bluedot-interpretability',
        config={
            'dataset': 'MNIST',
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'width': width,
            'depth': depth,
        }
    )

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, output_size, width, depth, activation):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, width)
        self.fcmiddle = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )
        self.fc2 = nn.Linear(width, output_size)

        self.activation = activation.lower()
        if self.activation != 'relu':
            raise ValueError('Currently, only ReLU activation is supported.')
        
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth

    def forward(self, x):
        intermediates = []
        x = F.relu(self.fc1(x))
        intermediates.append(x)
        for layer in self.fcmiddle:
            x = F.relu(layer(x))
            intermediates.append(x)
        x = self.fc2(x)
        return x, intermediates

# Instantiate the model and move it to the device
model = MLP(input_size, output_size, width, depth, activation).to(device)

# Transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the training and test datasets for MNIST
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)
criterion = nn.CrossEntropyLoss()

# Data loaders
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calculate_accuracy(y_pred, y_true):
    """
    Calculates the accuracy of the model's predictions.

    Args:
        y_pred (torch.Tensor): The raw output logits from the model (before activation).
        y_true (torch.Tensor): The true labels.

    Returns:
        float: The accuracy as a fraction between 0 and 1.
    """
    # Get the predicted class by finding the index with the maximum logit
    _, predicted = torch.max(y_pred, 1)
    # Compare predictions with true labels
    correct = (predicted == y_true).sum().item()
    # Calculate accuracy
    return correct / len(y_true)

def test_model(model, test_loader, criterion):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function used for evaluation.

    Returns:
        tuple: A tuple containing average test loss and average test accuracy.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    corrects = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, labels in test_loader:
            # Move images and labels to the device (CPU or GPU)
            images, labels = images.to(device), labels.to(device)
            # Flatten images if they are not already flattened
            images = images.view(images.size(0), -1)
            # Forward pass
            outputs, _ = model(images)
            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate accuracy
            accuracy = calculate_accuracy(outputs, labels)
            corrects += accuracy * len(labels)
            total += len(labels)

    # Calculate average loss and accuracy
    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = corrects / total

    return avg_test_loss, avg_test_accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, epoch_nbr):
    """
    Trains the model using the training data and evaluates it on the test data.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epoch_nbr (int): Number of epochs to train the model.

    Returns:
        dict: A dictionary containing the final training loss, training accuracy, test loss, and test accuracy.
    """
    for epoch in range(epoch_nbr):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for images, labels in train_loader:
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)
            # Flatten images
            images = images.view(images.size(0), -1)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, _ = model(images)
            # Compute loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            accuracy = calculate_accuracy(outputs, labels)
            running_corrects += accuracy * len(labels)
            total += len(labels)

        # Calculate average training loss and accuracy
        train_loss = running_loss / len(train_loader)
        train_accuracy = running_corrects / total

        # Evaluate the model on the test data
        test_loss, test_accuracy = test_model(model, test_loader, criterion)

        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })

        # Print progress every 10% of the total epochs or every epoch if total epochs < 10
        if (epoch + 1) % max(1, epoch_nbr // 10) == 0:
            print(f'Epoch [{epoch + 1}/{epoch_nbr}]')
            print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}')
            print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

    return {
        'train_loss': train_loss,
        'train_acc': train_accuracy,
        'test_loss': test_loss,
        'test_acc': test_accuracy
    }

def get_chatgpt_response(input_text, input_image=None):
    """
    Sends a request to the OpenAI ChatGPT API with the provided text and optional image,
    and returns the assistant's response.

    Args:
        input_text (str): The text prompt to send to the assistant.
        input_image (str, optional): Base64-encoded string of the image to send.

    Returns:
        str: The assistant's response text.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Construct the payload for the API request
    payload = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant attempting to interpret the functioning of a neural network "
                    "which may or may not be capable."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text
                    }
                ]
            }
        ]
    }
    
    # Include the image in the payload if provided
    if input_image is not None:
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{input_image}"
            }
        }
        payload['messages'][1]['content'].append(image_content)
    
    # Send the request and handle potential rate limiting
    responded = False
    wait_time = 0.5  # Initial wait time before retrying
    while not responded:
        # Send the POST request to the OpenAI API
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        try:
            # Parse the assistant's response from the API response
            message = response.json()['choices'][0]['message']['content']
            responded = True
        except KeyError:
            # Handle errors (e.g., rate limiting, invalid API key)
            print('Key error in producing ChatGPT message! Response:')
            print(response.json())
            print(f'Waiting {wait_time}s before trying again.')
            time.sleep(wait_time)
            wait_time *= 2  # Exponential backoff
    return message


# Function to get important weights connected to a specific output neuron
def get_important_weights(param, output_index):
    """
    Retrieves the most significant weights connected to a specific output neuron,
    sorted by their absolute values in descending order.

    Args:
        param (torch.nn.parameter.Parameter): The weight matrix of shape (output_size, input_size).
        output_index (int): The index of the output neuron.

    Returns:
        dict: A dictionary where keys are input indices and values are the weights as Python floats,
              limited to the top 10 most significant weights.
    """
    # Detach the parameter and move it to CPU if necessary
    weights = param.detach().cpu()
    # Extract the weights for the specified output index
    weights_for_output = weights[output_index]
    # Get absolute values for sorting
    abs_weights = torch.abs(weights_for_output)
    # Get sorted indices in descending order of importance
    sorted_indices = torch.argsort(-abs_weights)
    # Build the dictionary
    weight_dict = {}
    count = 1
    # The maximum number of weights to return
    max_weights = 10
    for idx in sorted_indices:
        index = idx.item()
        weight = weights_for_output[index].item()
        weight_dict[index] = weight
        if count >= max_weights:
            break
        count += 1
    return weight_dict

# Function to evaluate a specific feature (neuron) in the model
def evaluate_feature(model, depth_index, width_index, activation_stats, previous_layer=None, previous_weights=None):
    """
    Evaluates a specific neuron in the network by generating a prompt for GPT to interpret its functionality.

    Args:
        model (nn.Module): The neural network model.
        depth_index (int): The depth index of the neuron in the network.
        width_index (int): The index of the neuron within its layer.
        activation_stats (list): Activation statistics for neurons.
        previous_layer (str, optional): Summary of the previous layer's functionality.
        previous_weights (torch.nn.parameter.Parameter, optional): Weights of the previous layer.

    Returns:
        tuple: A message containing the GPT interpretation and the trigger input tensor.
    """
    stats = activation_stats[depth_index][width_index]
    dead_neuron = True
    # Check if the neuron is "dead" (not activating)
    for k in stats:
        if stats[k]['mean'] != 0:
            dead_neuron = False
            break
    if dead_neuron:
        print(f'Layer {depth_index} neuron {width_index} done and is dead')
        return 'Dead neuron', []

    # Construct the content prompt for GPT
    content_prompt = (
        gpt_prompt_base +
        'Your task is to evaluate what the following activation does.\n' +
        f'This activation is at depth {depth_index}, with the first layer outputs being labelled as depth 0.\n'
    )
    content_prompt += 'The following are the average and standard deviations of the activations for each output class:\n'
    for k in stats:
        content_prompt += f"Avg label {k}: {stats[k]['mean']}, std: {stats[k]['std']}\n"

    if previous_layer is not None:
        important_weights = get_important_weights(previous_weights, width_index)
        content_prompt += 'The previous layer was analysed as working in the following way:\n'
        content_prompt += previous_layer
        content_prompt += 'The most significant weights input to this neuron are:\n'
        for index, weight in important_weights.items():
            content_prompt += f'Activation {index}, weight: {weight}\n'

    content_prompt += (
        "The following image is an input specifically designed to maximise the output of the neuron.\n"
        "Please give an interpretation of what you think this feature means, and use reasoning based on the data and the image given.\n"
        "The interpretation must be precise enough to be able to accurately reconstruct which images would trigger this neuron, e.g., 'This neuron activates for the 90 degrees of curve from 0 to 90 degrees.'\n"
    )

    if previous_layer is not None:
        content_prompt += 'Connect this interpretation to the connections informed by the previous layer.\n'

    # Generate an input that maximizes the neuron's activation
    trigger_input = find_trigger_input(depth_index, width_index, model)
    image_file = get_image_file(trigger_input.cpu().detach())

    # Get the GPT response
    message_complete = False
    while not message_complete:
        try:
            message = get_chatgpt_response(content_prompt, image_file)
            message_complete = True
        except JSONDecodeError:
            print('JSON decode error, trying again')

    print(f'Layer {depth_index} neuron {width_index} done')
    return message, trigger_input

# Function to summarize the interpretations of all neurons in a layer
def summarise_layer(messages, layer_nbr):
    """
    Summarizes the interpretations of all neurons in a specific layer.

    Args:
        messages (list): List of interpretations for each neuron in the layer.
        layer_nbr (int): The layer number.

    Returns:
        str: A summary message for the entire layer.
    """
    content_prompt_start = (
        gpt_prompt_base +
        f'You have attempted to interpret each of the activations in layer {layer_nbr}.\n'
    )
    content_prompt = content_prompt_start + 'The following is the summary of each activation that you gave:\n'
    for i in range(len(messages)):
        content_prompt += f'Feature {i}: \n{messages[i]}\n\n\n'

    content_prompt += (
        'Now please give a summary of the most important features in the layer, feature by feature, and include all the details critical to predicting its activation. '
        'Feel free to ignore unimportant neurons such as dead neurons, and please include connections to previous layers in later layers.\n'
        'Example:\n'
        'Feature 1: Captures the horizontal top of digits, activating for digits such as 5 and 7 but not for the digit 4.\n'
        'Feature 2: Captures the circular nature of digits such as 0, made by positive weights from features 4 (left curve on right side of image) and 6 (right curve on left side of image) from the previous layer. '
        'It also has negative interference from features 3 and 7 (left curve on left side of image and right curve on right side of the image).\n'
        'Feature 4: Captures the absence of a center, made by a negative weight to feature 1 from layer 0 (presence of a center), which is amplified by other features such as a negative weight to feature 3, which captures the presence of a solid line down the middle.\n'
    )
    message = get_chatgpt_response(content_prompt)
    print(f'Layer {layer_nbr} summary:')
    print(message)
    return message

# Function to summarize the entire network's functionality
def summarise_network(layer_summaries, final_layer):
    """
    Summarizes how the neural network as a whole functions, based on layer summaries and the final layer weights.

    Args:
        layer_summaries (list): Summaries of each layer's functionality.
        final_layer (torch.nn.parameter.Parameter): The weight matrix of the final layer.

    Returns:
        str: A comprehensive summary of the network's overall operation.
    """
    content_prompt_start = (
        gpt_prompt_base +
        'You have summarized the functionality of each layer of the neural network.\n'
    )
    content_prompt = content_prompt_start + 'Here they are:\n'
    for i in range(len(layer_summaries)):
        content_prompt += f'Layer {i}:\n{layer_summaries[i]}\n'

    content_prompt += 'The most important connections for the final layer are:\n'
    for i in range(output_size):
        important_weights = get_important_weights(final_layer, i)
        content_prompt += f'For output {i}:\n'
        for index, weight in important_weights.items():
            content_prompt += f'Activation {index}, weight: {weight}\n'

    content_prompt += (
        "Concisely and precisely summarize how the network as a whole works, including specific influences of previous layers on later ones. "
        "Make sure to specify how each possible final output is decided upon from the penultimate layer.\n"
        "Example for an XOR network with 1 layer of 4 neurons:\n"
        "Summary: Feature 1 and 2 activate when the first and second inputs are active respectively, while Feature 0 doesn't contribute and Feature 3 increases with the total number of 1s. "
        "The final output is formed by activating the output if either feature 1 or 2 is active, and suppressing it if feature 3 is active."
    )

    summary = get_chatgpt_response(content_prompt)
    print(summary)
    return summary

def get_image_file(array_28x28):
    """
    Converts a flattened 28x28 array into a PNG image file and returns its base64 encoding.

    Args:
        array_28x28 (torch.Tensor): A 1D tensor of size 784 representing the image pixels.

    Returns:
        str: Base64 encoded string of the image file.
    """
    # Rescale the values to be between 0 and 255 for image representation
    rescaled_array = ((array_28x28.numpy() + 1) * 127.5).astype(np.uint8)
    # Reshape the array into a 28x28 image
    reshaped_array = rescaled_array.reshape(28, 28)
    
    # Convert the array to an image
    image = Image.fromarray(reshaped_array)
    
    # Save the image as a PNG file
    image_path = 'mnist_image.png'
    image.save(image_path)
    
    # Function to encode the image file to a base64 string
    def encode_image(image_path):
        """
        Encodes an image file to a base64 string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get the base64 encoded string of the image
    base64_image = encode_image(image_path)
    
    # Optionally, remove the temporary image file to free up space
    # os.remove(image_path)
    
    return base64_image

def find_trigger_input(depth_index, width_index, model):
    """
    Generates an input that maximizes the activation of a specific neuron in the network.

    Args:
        depth_index (int): The depth index of the neuron in the network.
        width_index (int): The index of the neuron within its layer.
        model (nn.Module): The neural network model.

    Returns:
        torch.Tensor: The input tensor that maximizes the neuron's activation.
    """
    model = model.to(device)
    started = False
    attempts = 0
    max_attempts = 10  # Prevent infinite loops

    while not started and attempts < max_attempts:
        # Initialize a random input tensor with gradient tracking
        input_img = torch.randn(model.input_size, requires_grad=True, device=device)
        
        # Define an optimizer to update the input tensor
        optimizer = optim.Adam([input_img], lr=0.1)
        
        # Forward pass through the model
        output, intermediate = model(input_img)
        
        # Check if the neuron is activating
        if intermediate[depth_index][width_index] != 0:
            started = True
        attempts += 1

    if not started:
        print(f"Neuron at depth {depth_index}, width {width_index} did not activate after {max_attempts} attempts.")
        return input_img  # Return the last input_img even if the neuron did not activate

    # Optimize the input to maximize the neuron's activation
    for i in range(200):
        optimizer.zero_grad()
        _, intermediate = model(input_img)
        # Define the loss as the negative of the neuron's activation plus a regularization term
        loss = -intermediate[depth_index][width_index] + torch.sum(torch.square(input_img))
        loss.backward()
        optimizer.step()

    return input_img

def draw_image(arr):
    """
    Displays an image represented by a flattened array using matplotlib.

    Args:
        arr (torch.Tensor): A 1D tensor representing the image pixels.
    """
    # Convert the tensor to a NumPy array
    new_arr = arr.cpu().detach().numpy()
    # Reshape the array into a 28x28 image
    pixels = new_arr.reshape((28, 28))
    # Display the image using matplotlib
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()

def make_test_set(model, num_samples=100):
    """
    Creates datasets of correctly and incorrectly classified samples by the model.

    Args:
        model (nn.Module): The trained neural network model.
        num_samples (int, optional): The number of correct and incorrect samples to collect. Defaults to 100.

    Returns:
        tuple:
            - correct_dataset (list): A list containing correctly classified images and their labels.
            - incorrect_dataset (list): A list containing incorrectly classified images and their true labels.
            - incorrect_answers (list): A list of the model's incorrect predictions for the incorrectly classified images.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Lists to store images and labels
    correct_images = []
    correct_labels = []
    incorrect_images = []
    incorrect_labels = []
    incorrect_answers = []
    
    with torch.no_grad():
        loaders = [test_loader, train_loader]  # Use both test and train data
        for data_loader in loaders:
            for images, labels in data_loader:
                # Move images and labels to the device
                images, labels = images.to(device), labels.to(device)
                
                # Flatten images if necessary
                images = images.view(images.size(0), -1)
                # Get model predictions
                outputs, _ = model(images)
                _, preds = torch.max(outputs, 1)
                
                # Compare predictions to actual labels
                correct_mask = preds == labels
                incorrect_mask = preds != labels
                
                # Append correct and incorrect images and labels
                correct_images.extend(images[correct_mask].cpu())
                correct_labels.extend(labels[correct_mask].cpu())
                incorrect_images.extend(images[incorrect_mask].cpu())
                incorrect_labels.extend(labels[incorrect_mask].cpu())
                incorrect_answers.extend(preds[incorrect_mask].cpu())
                
                # Stop once enough samples are collected
                if len(correct_images) >= num_samples and len(incorrect_images) >= num_samples:
                    break
            else:
                continue  # Only executed if the inner loop did NOT break
            break  # Break the outer loop if enough samples are collected

    # Trim lists to exactly the number of requested samples
    correct_images = correct_images[:num_samples]
    correct_labels = correct_labels[:num_samples]
    incorrect_images = incorrect_images[:num_samples]
    incorrect_labels = incorrect_labels[:num_samples]
    incorrect_answers = incorrect_answers[:num_samples]

    # Combine correct and incorrect samples into separate datasets
    correct_dataset = [correct_images, correct_labels]
    incorrect_dataset = [incorrect_images, incorrect_labels]
    
    return correct_dataset, incorrect_dataset, incorrect_answers

def test_interpretation(network_summary, layer_summaries, model):
    """
    Evaluates the network's interpretability by comparing the model's predictions with GPT's interpretations.

    Args:
        network_summary (str): A summary of the network's overall functionality.
        layer_summaries (list): A list of summaries for each layer in the network.
        model (nn.Module): The trained neural network model.

    Returns:
        None
    """
    def get_interpretation_answer(image):
        """
        Generates a prompt based on the network summaries and gets GPT's prediction for a given image.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            int: The predicted label extracted from GPT's response, or -1 if parsing fails.
        """
        # Construct the prompt for GPT
        prompt = gpt_prompt_base + 'You have evaluated the layers of the network as follows:\n'
        for layer in range(len(layer_summaries)):
            prompt += f'Layer {layer}:\n{layer_summaries[layer]}'
        prompt += 'You summarised the network as working overall as follows:\n'
        prompt += network_summary
        prompt += (
            'Now, thinking step by step, predict what the network will say the following image is. '
            'Ensure that the final character of your response is the digit you would like to answer with, and note that for '
            'half of the images you are asked about, the network prediction will be incorrect.\n'
        )
        # Get GPT's response using the prompt and the image
        response = get_chatgpt_response(prompt, get_image_file(image))
        try:
            # Extract the last digit from the response
            return int(''.join(x for x in response[-20:] if x.isdigit())[-1])
        except (ValueError, IndexError):
            print('Error: GPT did not respond with a valid integer.')
            print('Response:', response)
            return -1

    # Generate test datasets
    correct_dataset, incorrect_dataset, incorrect_answers = make_test_set(model)
    print('Test set made')

    # Initialize counters for tracking prediction outcomes
    doubly_correct_predictions = 0  # Both GPT and network are correct
    gpt_incorrect_network_correct = 0  # GPT is incorrect, network is correct
    network_incorrect_gpt_correct_about_network = 0  # GPT predicts network's incorrect output
    network_incorrect_gpt_correct_about_image = 0  # GPT predicts the correct label despite network's error
    doubly_incorrect_predictions = 0  # Both GPT and network are incorrect

    # Evaluate GPT's predictions on correctly classified images
    for image_index in range(len(correct_dataset[0])):
        gpt_answer = get_interpretation_answer(correct_dataset[0][image_index])
        correct_label = correct_dataset[1][image_index].item()
        if gpt_answer == correct_label:
            doubly_correct_predictions += 1
        else:
            gpt_incorrect_network_correct += 1

        # Progress logging
        if len(correct_dataset[0]) < 10 or (image_index + 1) % (len(correct_dataset[0]) // 10) == 0:
            print(f'Predictions for image {image_index + 1} of correct dataset done.')

    # Evaluate GPT's predictions on incorrectly classified images
    for image_index in range(len(incorrect_dataset[0])):
        gpt_answer = get_interpretation_answer(incorrect_dataset[0][image_index])
        correct_label = incorrect_dataset[1][image_index].item()
        model_label = incorrect_answers[image_index].item()
        if gpt_answer == model_label:
            network_incorrect_gpt_correct_about_network += 1
        elif gpt_answer == correct_label:
            network_incorrect_gpt_correct_about_image += 1
        else:
            doubly_incorrect_predictions += 1

        # Progress logging
        if len(incorrect_dataset[0]) < 10 or (image_index + 1) % (len(incorrect_dataset[0]) // 10) == 0:
            print(f'Predictions for image {image_index + 1} of incorrect dataset done.')

    tot_correct_images = len(correct_dataset[0])
    tot_incorrect_images = len(incorrect_dataset[0])

    # Print evaluation results
    print('Fraction correct when network correct:', doubly_correct_predictions / tot_correct_images)
    print('Fraction incorrect when network correct:', gpt_incorrect_network_correct / tot_correct_images)
    print('Fraction correct about network when network incorrect:', network_incorrect_gpt_correct_about_network / tot_incorrect_images)
    print('Fraction correct about image when network incorrect:', network_incorrect_gpt_correct_about_image / tot_incorrect_images)
    print('Fraction incorrect about both network and image:', doubly_incorrect_predictions / tot_incorrect_images)
    
def get_activations(model, train_loader):
    """
    Collects activation statistics (mean and standard deviation) for each neuron in the model across different classes.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for the training dataset.

    Returns:
        list: A nested list containing activation statistics for each neuron in each layer.
              The structure is [layer][neuron][class] = {'mean': ..., 'std': ..., 'occurrence_count': ...}
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize the activations list
    # Structure: activations[layer][neuron][class_label] = {'tot output': ..., 'tot output squared': ..., 'occurrence count': ...}
    activations = [
        [
            {
                i: {'tot output': 0, 'tot output squared': 0, 'occurrence count': 0}
                for i in range(output_size)
            }
            for _ in range(width)
        ]
        for __ in range(depth)
    ]

    with torch.no_grad():
        imgs_counted = 0
        for images, labels in train_loader:
            # Move images and labels to CPU
            images, labels = images.cpu(), labels.cpu()
            # Flatten images
            images = images.view(images.size(0), -1)
            # Ensure the model is on CPU
            model = model.cpu()
            for image, label in zip(images, labels):
                imgs_counted += 1
                # Forward pass through the model
                output, features = model(image)
                # Iterate over layers and neurons to accumulate activations
                for i in range(len(activations)):
                    for j in range(len(activations[i])):
                        class_label = int(label)
                        neuron_output = features[i][j].item()
                        activations[i][j][class_label]['tot output'] += neuron_output
                        activations[i][j][class_label]['tot output squared'] += neuron_output ** 2
                        activations[i][j][class_label]['occurrence count'] += 1
                # Limit the number of images processed for efficiency
                if imgs_counted >= 1000:
                    break
            else:
                continue  # Only executed if the inner loop did NOT break
            break  # Break the outer loop if enough images are processed

        # Calculate mean and standard deviation for each neuron and class
        for i in range(len(activations)):
            for j in range(len(activations[i])):
                for k in activations[i][j]:
                    stats = activations[i][j][k]
                    tot = stats['tot output']
                    nbr = stats['occurrence count']
                    squ = stats['tot output squared']
                    if nbr > 0:
                        # Calculate mean
                        mean = tot / nbr
                        # Calculate standard deviation
                        std = np.sqrt(squ / nbr - (mean) ** 2)
                        activations[i][j][k]['mean'] = mean
                        activations[i][j][k]['std'] = std
                    else:
                        # Handle the case with zero occurrences
                        activations[i][j][k]['mean'] = 0
                        activations[i][j][k]['std'] = 0
    return activations

def main():
    """
    The main function orchestrates the training, activation analysis, feature evaluation,
    layer summarization, network summarization, and interpretation testing.
    """
    global gpt_prompt_base

    # Initialize messages and layer summaries
    messages = [['' for _ in range(width)] for _ in range(depth)]
    layer_summaries = [None]

    print('Training MNIST model')

    # Train the model and get performance metrics
    performance = train_model(model, train_loader, test_loader, criterion, optimizer, epoch_nbr)

    # Update the GPT prompt with performance metrics
    gpt_prompt_base += (
        f"The train accuracy of the final model is {performance['train_acc']}, "
        f"the train loss is {performance['train_loss']}. "
        f"The test accuracy of the final model is {performance['test_acc']}, "
        f"the test loss is {performance['test_loss']}."
    )

    # Get activation statistics for the neurons
    activation_stats = get_activations(model, train_loader)

    # Collect parameters (weights) from the model
    params = []
    for name, param in model.named_parameters():
        if name.endswith('weight'):
            params.append(param)

    # Evaluate each neuron and summarize layers
    for i in range(depth):
        for j in range(width):
            # Evaluate the feature (neuron) and get interpretation
            message, trigger_input = evaluate_feature(
                model, i, j, activation_stats, layer_summaries[-1], params[i - 1] if i > 0 else None
            )
            messages[i][j] = message
        # Summarize the layer based on neuron messages
        layer_summaries.append(summarise_layer(messages[i], i))

    # Summarize the entire network
    summary = summarise_network(layer_summaries[1:], params[-1])

    # Test the interpretation of the network
    test_interpretation(summary, layer_summaries[1:], model)

# Run the main function
if __name__ == '__main__':
    main()
