import os
import openai
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import requests

# Initialize OpenAI API
# It's recommended to store your API key securely, e.g., using environment variables
# Get an openai api key from https://openai.com/api/ and then set an environment variable called "OPENAI_API_KEY" and set it to the key
api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device being used: {device}')


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
