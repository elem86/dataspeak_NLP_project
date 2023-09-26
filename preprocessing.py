import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm

# Load the dataset
data_path = "QA_cleaned.csv"
data = pd.read_csv(data_path)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)


# Function to find start and end positions of the answer in the context
def find_start_end_positions(context, answer):
    # Tokenize context and answer
    context_tokens = tokenizer.tokenize(context)
    answer_tokens = tokenizer.tokenize(answer)

    # Search for the start and end tokens of the answer in the context tokens
    for i in range(len(context_tokens) - len(answer_tokens) + 1):
        if context_tokens[i : i + len(answer_tokens)] == answer_tokens:
            return i, i + len(answer_tokens) - 1  # Return start and end positions

    # If answer not found in context, return positions pointing to [CLS] token
    return 0, 0


# Calculate start and end positions
start_positions = []
end_positions = []
for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
    context = row["BERT_Context"]  # Corrected to use the BERT_Context column
    answer = row["Cleaned_Body_answer"]  # Assuming this column contains the answer

    # Check if context and answer are strings, if not, replace with empty string or skip
    if not isinstance(context, str) or not isinstance(answer, str):
        start_positions.append(0)
        end_positions.append(0)
        continue  # skip to the next row

    start_idx, end_idx = find_start_end_positions(context, answer)
    start_positions.append(start_idx)
    end_positions.append(end_idx)

# Add start and end positions to the dataframe
data["Start_Positions"] = start_positions
data["End_Positions"] = end_positions

# Now, cast to integer
data["Start_Positions"] = data["Start_Positions"].astype(int)
data["End_Positions"] = data["End_Positions"].astype(int)

# Save the preprocessed dataframe
data.to_csv("preprocessed_data.csv", index=False)
