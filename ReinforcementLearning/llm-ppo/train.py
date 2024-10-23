import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam

# Initialize model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
active_model = GPT2LMHeadModel.from_pretrained(model_name)
ref_model = GPT2LMHeadModel.from_pretrained(model_name).eval()  # Reference model (frozen for evaluation)

# Optimizer setup
optimizer = Adam(active_model.parameters(), lr=5e-5)


# Function to compute advantages (for policy gradient)
def compute_advantages(values, rewards):
    """
    Compute advantages and returns for PPO.

    Args:
        values (Tensor): Predicted state values by the model.
        rewards (Tensor): Actual rewards received from the environment.

    Returns:
        advantages (Tensor): The advantages for each token position.
        returns (Tensor): Discounted rewards used for value loss calculation.
    """
    returns = rewards + values
    advantages = returns - values
    return advantages, returns


# PPO Loss calculation
def ppo_loss(logprobs, old_logprobs, advantages, returns, values, clip_range=0.2):
    """
    Compute the PPO loss function.

    Args:
        logprobs (Tensor): Log probabilities of actions by the current policy.
        old_logprobs (Tensor): Log probabilities from the reference model (old policy).
        advantages (Tensor): Pre-computed advantages.
        returns (Tensor): The returns for value loss calculation.
        values (Tensor): State value predictions from the model.
        clip_range (float): Clipping range for PPO to avoid excessive updates.

    Returns:
        loss (Tensor): Total PPO loss including policy loss and value function loss.
    """
    # Compute probability ratios
    ratio = torch.exp(logprobs - old_logprobs)

    # Compute clipped policy loss
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss = torch.max(pg_losses, pg_losses2)

    # Value function loss (clipped)
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (torch.clamp(values, values - clip_range, values + clip_range) - returns) ** 2
    vf_loss = torch.max(vf_loss1, vf_loss2)

    # Combine policy loss and value function loss
    loss = pg_loss.mean() + 0.5 * vf_loss.mean()
    return loss


# Training loop
def train(prompt_dataset, batch_size=2, epochs=3):
    """
    Train the active model using PPO on a given prompt dataset.

    Args:
        prompt_dataset (list): List of tokenized prompt sequences.
        batch_size (int): The size of each training batch.
        epochs (int): Number of epochs to train on each batch.

    Returns:
        None
    """
    for batch_prompt in prompt_dataset:
        # Step 1: Generate responses from the active model
        # Input: `batch_prompt` with shape (batch_size, sequence_length)
        batch_response = active_model.generate(batch_prompt)  # Generates a response for each prompt

        # Step 2: Concatenate prompts with responses to create the final input for training
        batch_data = torch.cat((batch_prompt, batch_response), dim=1)  # Shape: (batch_size, full_sequence_length)

        # Step 3: Forward pass through reference model (ref_model) to get old log_probs and values
        with torch.no_grad():
            ref_outputs = ref_model(batch_data)  # Reference outputs from frozen model
            ref_all_probs = torch.log_softmax(ref_outputs.logits, dim=-1)  # Log probabilities
            ref_all_values = ref_outputs.logits.mean(dim=-1)  # Approximation for value prediction

        # Step 4: Forward pass through active model to get log_probs and values
        active_outputs = active_model(batch_data)  #(batch_size, sequence_length, vocab_size)       Active model's forward pass
        # The active_model outputs logits
        #  logits: These are the unnormalized predictions (before softmax) for each token at each position in the sequence.
        # They represent the raw prediction scores for each token in the vocabulary. For instance,
        # if using GPT-2 with a vocabulary size of 50,257, the logits tensor will predict the likelihood of each of these 50,257 tokens being the next token in the sequence.
        batch_all_probs = torch.log_softmax(active_outputs.logits, dim=-1)  # (batch_size, sequence_length, vocab_size)     Active log probabilities
        batch_all_values = active_outputs.logits.mean(dim=-1)  # (batch_size, sequence_length)      Approximation for value prediction,

        # Step 5: Compute rewards (customize according to the task)
        rewards = compute_rewards(batch_data)  # Use a custom reward function here

        # Step 6: Compute advantages and returns
        advantages, returns = compute_advantages(batch_all_values, rewards)

        # Inner training loop (for each batch, run multiple epochs)
        for _ in range(epochs):
            # Active model forward pass to recompute log_probs and values
            active_outputs = active_model(batch_data)
            all_probs = torch.log_softmax(active_outputs.logits, dim=-1)
            all_values = active_outputs.logits.mean(dim=-1)

            # Compute the PPO loss
            loss = ppo_loss(all_probs, ref_all_probs, advantages, returns, all_values)

            # Backpropagation and optimization
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients

        print(f"Training completed for a batch of size: {batch_size}")


# Custom reward function (this needs to be defined for your specific task)
def compute_rewards(batch_data):
    """
    Example reward function that simply rewards based on the length of generated text.
    Customize this function based on your task's specific requirements.

    Args:
        batch_data (Tensor): Concatenated prompts and responses (tokenized).

    Returns:
        rewards (Tensor): Rewards for each token in the batch.
    """
    # Example: reward based on the number of tokens in the response (customize this)
    rewards = torch.tensor([len(text) for text in batch_data])
    return rewards


# Example dataset
prompt_dataset = ["Once upon a time", "In a galaxy far away", "The quick brown fox"]

# Tokenize the dataset using GPT2 tokenizer
prompt_dataset = [torch.tensor(tokenizer.encode(prompt)) for prompt in prompt_dataset]

# Train the model using PPO
train(prompt_dataset, batch_size=2, epochs=3)
