def generate_text_from_label(label, label_map, tokenizer, model, max_length=100):
    """
    Generate text conditioned on a given label using label embeddings.

    Args:
    - label: The label to condition on (e.g., "building").
    - label_map: Dictionary mapping label names to label IDs.
    - tokenizer: The tokenizer used for encoding.
    - model: The trained LabelConditionedGPT2 model.
    - max_length: The maximum length of generated text.

    Returns:
    - Generated text as a string.
    """
    # Convert the label to an ID using the label_map
    label_id = torch.tensor([label_map[label]], dtype=torch.long).to(model.device)
    
    # Prepare an empty input to start generation (no initial text, just label conditioning)
    input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(model.device)

    # Generate text conditioned on the label embedding
    output_ids = model.generate(
        input_ids=input_ids,
        label_ids=label_id,
        max_length=max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Ensure that generation stops correctly
        do_sample=True,  # Use sampling for more varied output
        temperature=0.7,  # Control randomness
        top_k=50,  # Consider only top 50 words for sampling
        top_p=0.95  # Use nucleus sampling
    )
    
    # Decode the generated output to text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


# Load the tokenizer and model
m3_tokenizer = PreTrainedTokenizerFast.from_pretrained("trained_tokenizer")
model = LabelConditionedGPT2.from_pretrained("trained_model/final", num_labels=len(label_map))
model.to('cuda')  # Move to GPU if available

# Example label to generate text for
label = "building"

# Generate text for a given label
generated_text = generate_text_from_label(label, label_map, m3_tokenizer, model, max_length=100)

print(f"Generated Text for label '{label}':\n{generated_text}")
