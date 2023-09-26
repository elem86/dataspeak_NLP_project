import pandas as pd
from transformers import BertTokenizer
import random


# Load the preprocessed dataframe
data = pd.read_csv("preprocessed_data.csv")

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Number of rows to check
num_rows_to_check = 10  # You can change this to the number of rows you want to check

# Randomly select rows to check
rows_to_check = random.sample(range(len(data)), num_rows_to_check)

for row_idx in rows_to_check:
    row = data.iloc[row_idx]

    # Extract context, answer, start and end positions
    context = row["BERT_Context"]
    answer = row["Cleaned_Body_answer"]
    start_pos = row["Start_Positions"]
    end_pos = row["End_Positions"]

    # Tokenize context and extract the answer using start and end positions
    context_tokens = tokenizer.tokenize(context)
    extracted_answer_tokens = context_tokens[start_pos : end_pos + 1]
    extracted_answer = tokenizer.convert_tokens_to_string(extracted_answer_tokens)

    # Print the results for manual verification
    print(f"Row:  {row_idx}")
    print(f"Context:  {' '.join(context_tokens)}")
    print(f"Answer:  {' '.join(tokenizer.tokenize(answer))}")
    print(f"Extracted Answer:  {' '.join(extracted_answer_tokens)}")
    print("-----")
