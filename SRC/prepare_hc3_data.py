from datasets import load_dataset
import pandas as pd

print("Loading HC3 dataset from Hugging Face...")

# Load the English subset of HC3
dataset = load_dataset("Hello-SimpleAI/HC3", name="all")

print(dataset)

# We will use the train split if available
# and create a small balanced dataset from it.
rows = []

# Limit size to keep things simple and fast
max_human = 1000
max_ai = 1000

human_count = 0
ai_count = 0

for item in dataset["train"]:
    # HC3 examples contain human answers and ChatGPT answers
    # We collect both sides into the same simple table

    # Human answers
    human_answers = item.get("human_answers", [])
    for text in human_answers:
        if isinstance(text, str) and text.strip():
            rows.append({
                "text": text.strip(),
                "label": "human"
            })
            human_count += 1
            if human_count >= max_human:
                break

    # ChatGPT answers
    chatgpt_answers = item.get("chatgpt_answers", [])
    for text in chatgpt_answers:
        if isinstance(text, str) and text.strip():
            rows.append({
                "text": text.strip(),
                "label": "ai"
            })
            ai_count += 1
            if ai_count >= max_ai:
                break

    if human_count >= max_human and ai_count >= max_ai:
        break

df = pd.DataFrame(rows)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = "data/processed/hc3_human_ai.csv"
df.to_csv(output_path, index=False)

print(f"Saved cleaned dataset to: {output_path}")
print(df["label"].value_counts())
print(df.head())