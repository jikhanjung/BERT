from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load GeoBERT Model
MODEL_NAME = "botryan96/GeoBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, from_tf=True)

# Define NER Pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example text
text = "The Upper Cambrian fossil was found in Queensland, Australia, and Texas, USA."
ner_results = ner_pipeline(text)

# Function to merge subwords and align missing entity parts
def merge_and_align_entities(ner_results, text, tokenizer):
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
    
    merged_entities = []
    current_entity = {"word": "", "label": None, "start": None, "end": None}

    # Align recognized entities with full words
    for entity in ner_results:
        word = entity["word"]
        label = entity["entity"]
        start, end = entity["start"], entity["end"]

        # Handle subword merging
        if word.startswith("##"):
            current_entity["word"] += word[2:]  # Merge with previous word
            current_entity["end"] = end
        else:
            if current_entity["word"]:  # Append previous entity
                merged_entities.append(current_entity)
            current_entity = {"word": word, "label": label, "start": start, "end": end}  # Start new entity

    if current_entity["word"]:  # Append last entity
        merged_entities.append(current_entity)

    # Post-process: Ensure missing first subwords are included
    corrected_entities = []
    for entity in merged_entities:
        start, end = entity["start"], entity["end"]
        # Check if the entity's start position matches any token's start
        for token, (token_start, token_end) in zip(tokens, token_offsets):
            if token_start == start and not token.startswith("##"):
                entity["word"] = token + entity["word"]
                entity["start"] = token_start
                break
        corrected_entities.append(entity)

    return corrected_entities

# Apply entity alignment
fixed_entities = merge_and_align_entities(ner_results, text, tokenizer)

# Print fixed results
print("\n=== Fixed Named Entity Recognition (NER) Results ===")
for entity in fixed_entities:
    print(f"Entity: {entity['word']}, Label: {entity['label']}")
