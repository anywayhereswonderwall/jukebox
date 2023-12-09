from model import GPT
from config import GPTConfig, device
import numpy as np
import torch
conf = GPTConfig()
model = GPT(conf)

model.to(device)

model = torch.compile(model)
model.load_state_dict(torch.load('weights/model_weights_v2.pth', map_location=device))

model.eval()


def get_context(num_samples):
    # Set the mean and standard deviation for the normal distribution
    mean = 63.5  # (0 + 127) / 2
    std_dev = 21.17  # Adjust this value based on your preference

    # Generate random numbers from a normal distribution
    random_numbers = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

    # Round the numbers to integers and clip them to the range [0, 127]
    random_integers = np.clip(np.round(random_numbers), 0, 127).astype(int)

    return torch.tensor(random_integers).to(device).view(-1, 1)


def sample_from_model(max_tokens, temperature):
    """
    Params:
    max_tokens: maximum number of tokens to generate
    temperature: temperature for sampling distribution
    Returns:
    samples: a list of samples of MIDI encoded events (list of ints)
    """
    """
    Samples from a trained model.
    """
    NUM_SAMPLES = 1
    return torch.flatten(model.generate(get_context(NUM_SAMPLES), max_new_tokens=max_tokens, temperature=temperature))
