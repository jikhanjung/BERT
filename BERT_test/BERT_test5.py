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

# Function to merge subwords properly
def merge_entities(ner_results):
    merged_entities = []
    current_word = ""
    current_label = None

    for entity in ner_results:
        word = entity["word"]
        label = entity["entity"]

        # Merge subwords correctly
        if word.startswith("##"):
            current_word += word[2:]  # Remove ## and append
        else:
            if current_word:  # Store previous entity before starting a new one
                merged_entities.append({"word": current_word, "label": current_label})
            current_word = word  # Start new entity
            current_label = label

    # Add last entity
    if current_word:
        merged_entities.append({"word": current_word, "label": current_label})

    return merged_entities

# Apply entity merging
fixed_entities = merge_entities(ner_results)

# Print results
print("\n=== Corrected Named Entity Recognition (NER) Results ===")
for entity in fixed_entities:
    print(f"Entity: {entity['word']}, Label: {entity['label']}")
