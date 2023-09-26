# Dataspeak NLP project using BERT

## Data Cleaning Script

This Python script is designed to clean and preprocess datasets, particularly focusing on text data. It utilizes libraries like pandas for data manipulation, re for regular expressions, spacy for natural language processing tasks, and multiprocessing to parallelize the cleaning process, enhancing efficiency.

### Overview

The script performs the following tasks:
- Loads datasets and handles encoding issues.
- Cleans text data by removing URLs, HTML tags, non-alphanumeric characters, and performs lemmatization and stop word removal using Spacy.
- Applies cleaning functions in parallel to enhance efficiency.
- Converts columns to appropriate data types and handles missing values.
- Merges DataFrames to create a consolidated dataset.
- Saves the cleaned and processed datasets.

#### Dependencies

- pandas
- re
- spacy
- multiprocessing
- tqdm

#### How to Run

1. Ensure all dependencies are installed.
2. Place the script in the same directory as your datasets.
3. Run the script using a Python interpreter.

```bash
$ python data_cleaning.py
```

### Code Structure

#### Importing Libraries
```python
import pandas as pd
import re
import spacy
from multiprocessing import Pool
from tqdm import tqdm
```
#### Loading Spacy Model
```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
```

#### Defining Cleaning Function
```python
def clean_text(text):
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\r\n|\n|\t", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(words)
```

#### Parallel Cleaning
```python
def parallel_clean(df, column_name):
    with Pool() as pool:
        return list(tqdm(pool.imap(clean_text, df[column_name]), total=df.shape[0]))
```

#### Main Function
```python
def main():
    answers = pd.read_csv("Answers.csv", encoding="ISO-8859-1")
    questions = pd.read_csv("Questions.csv", encoding="ISO-8859-1")
    tags = pd.read_csv("Tags.csv", encoding="ISO-8859-1")
    
    # Preprocessing steps
    def preprocess(df):
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    
    answers = preprocess(answers)
    questions = preprocess(questions)
    tags = preprocess(tags)
    
    # Convert columns to appropriate data types
    answers["CreationDate"] = pd.to_datetime(answers["CreationDate"])
    questions["CreationDate"] = pd.to_datetime(questions["CreationDate"])
    answers["OwnerUserId"] = answers["OwnerUserId"].fillna(0).astype(int)
    questions["OwnerUserId"] = questions["OwnerUserId"].fillna(0).astype(int)
    
    # Applying cleaning 'Body' and 'Title' columns of questions and answers datasets in parallel
    questions["Cleaned_Body"] = parallel_clean(questions, "Body")
    answers["Cleaned_Body"] = parallel_clean(answers, "Body")
    questions["Cleaned_Title"] = parallel_clean(questions, "Title")
    
    # Drop unnecessary columns
    answers = answers.drop("Body", axis=1)
    questions = questions.drop(["Body", "Title"], axis=1)
    
    # Merging DataFrames
    tags = tags.groupby("Id")["Tag"].agg(", ".join).reset_index()
    questions_tags_merged = questions.merge(tags, left_on="Id", right_on="Id", how="left")
    qa_merged = questions_tags_merged.merge(answers, left_on="Id", right_on="ParentId", how="left", suffixes=("_question", "_answer"))
    
    # Drop unnecessary columns and rows
    qa_merged = qa_merged.dropna(subset=["Cleaned_Body_question", "Cleaned_Title", "Tag", "Cleaned_Body_answer"])
    qa_merged = qa_merged.reset_index(drop=True)
    
    # Create BERT_Context by combining cleaned body of the question and body of the answer
    qa_merged["BERT_Context"] = qa_merged["Cleaned_Body_question"] + " " + qa_merged["Cleaned_Body_answer"]
    
    # Saving the cleaned datasets
    qa_merged.to_csv("QA_cleaned.csv", index=False)
    print("QA_cleaned.csv saved!")
    answers.to_csv("Answers_cleaned.csv", index=False)
    print("Answers_cleaned.csv saved!")
    questions.to_csv("Questions_cleaned.csv", index=False)
    print("Questions_cleaned.csv saved!")
```

#### Execution
```python
if __name__ == "__main__":
    main()
```


## Preprocessing Script

### Overview
This script is responsible for preprocessing the QA_cleaned.csv dataset. It calculates the start and end positions of the answer in the context and adds these positions to the dataframe. The preprocessed data is then saved as preprocessed_data.csv.

#### Importing Libraries
```python
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
```

#### Loading Dataset and Initializing Tokenizer
```python
data_path = "QA_cleaned.csv"
data = pd.read_csv(data_path)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
```

#### Function to Find Start and End Positions
```python
def find_start_end_positions(context, answer):
    context_tokens = tokenizer.tokenize(context)
    answer_tokens = tokenizer.tokenize(answer)
    
    for i in range(len(context_tokens) - len(answer_tokens) + 1):
        if context_tokens[i : i + len(answer_tokens)] == answer_tokens:
            return i, i + len(answer_tokens) - 1
    
    return 0, 0
```

#### Calculating Start and End Positions
```python
start_positions = []
end_positions = []

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing rows"):
    context = row["BERT_Context"]
    answer = row["Cleaned_Body_answer"]
    
    if not isinstance(context, str) or not isinstance(answer, str):
        start_positions.append(0)
        end_positions.append(0)
        continue
    
    start_idx, end_idx = find_start_end_positions(context, answer)
    start_positions.append(start_idx)
    end_positions.append(end_idx)
```

#### Adding Positions to DataFrame and Saving
```python
data["Start_Positions"] = start_positions
data["End_Positions"] = end_positions

data["Start_Positions"] = data["Start_Positions"].astype(int)
data["End_Positions"] = data["End_Positions"].astype(int)

data.to_csv("preprocessed_data.csv", index=False)
```

#### Execution
To execute the script, simply run it in a Python environment that has access to the required libraries and the input dataset. The output will be saved as preprocessed_data.csv in the same directory as the script.



## Verifying Script

### Overview
This script is designed to verify the correctness of the preprocessed data by extracting answers from the context using the calculated start and end positions and comparing them with the actual answers. It randomly selects rows from the preprocessed data and prints the context, answer, and extracted answer for manual verification.

#### Importing Libraries
```python
import pandas as pd
from transformers import BertTokenizer
import random
```

#### Loading Preprocessed Data and Initializing Tokenizer
```python
data = pd.read_csv("preprocessed_data.csv")

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
```

#### Selecting Rows to Check and Extracting Answers
```python
num_rows_to_check = 10  # You can change this to the number of rows you want to check
rows_to_check = random.sample(range(len(data)), num_rows_to_check)

for row_idx in rows_to_check:
    row = data.iloc[row_idx]
    context = row["BERT_Context"]
    answer = row["Cleaned_Body_answer"]
    start_pos = row["Start_Positions"]
    end_pos = row["End_Positions"]
    
    context_tokens = tokenizer.tokenize(context)
    extracted_answer_tokens = context_tokens[start_pos : end_pos + 1]
    extracted_answer = tokenizer.convert_tokens_to_string(extracted_answer_tokens)
```

#### Printing Results for Manual Verification
```python
    print(f"Row:  {row_idx}")
    print(f"Context:  {' '.join(context_tokens)}")
    print(f"Answer:  {' '.join(tokenizer.tokenize(answer))}")
    print(f"Extracted Answer:  {' '.join(extracted_answer_tokens)}")
    print("-----")
```

#### Execution
To execute the script, simply run it in a Python environment that has access to the required libraries and the preprocessed data. The output will be printed to the console for manual verification.

## BERT Model Training Notebook

### Overview
This Jupyter notebook is designed to fine-tune a BERT model for Question Answering tasks. It involves loading a pre-trained BERT model, preparing the data, and training the model with the possibility of using mixed precision training. The notebook is structured to run on Google Colab and utilizes Google Drive for storing datasets and the fine-tuned model.

#### Installation of Libraries
```shell
!pip install -q transformers
```

#### Importing Libraries and Modules
```python
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from google.colab import drive
import os
```

#### Mounting Google Drive and Configuring Logging
```python
drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Dataspeak')

logging.basicConfig(
    filename="bert_training_log.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
```

#### Loading Model, Tokenizer, and Preprocessed Data
```python
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

preprocessed_data_path = "preprocessed_data.csv"
data = pd.read_csv(preprocessed_data_path)
```

#### Data Preparation and DataLoader Initialization
```python
# Split the data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)

# Initialize GradScaler for mixed precision training
scaler = GradScaler()

class QADataset(Dataset):
    def __init__(self, questions, contexts, start_positions, end_positions):
        self.questions = questions
        self.contexts = contexts
        self.start_positions = start_positions
        self.end_positions = end_positions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        if not isinstance(question, str) or not isinstance(context, str):
            logging.error(
                f"Invalid types - Question: {type(question)}, Context: {type(context)} at index {idx}"
            )
            return None
        try:
            inputs = tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length",  # Ensure sequences are padded to max_length
            )
        except Exception as e:
            logging.error(f"Error in tokenization at index {idx}: {e}")
            return None

        start_position = self.start_positions[idx]
        end_position = self.end_positions[idx]

        # Handle cases where model mispredicts and start_position is after end_position
        if start_position > end_position:
            start_position, end_position = end_position, start_position

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "start_positions": torch.tensor(start_position, dtype=torch.long),
            "end_positions": torch.tensor(end_position, dtype=torch.long),
        }


# Define your custom collate function here
def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None  # Return None if all items are filtered out
    return torch.utils.data.dataloader.default_collate(batch)


# Initialize the training and validation datasets
train_dataset = QADataset(
    questions=train_data["Cleaned_Title"].tolist(),
    contexts=train_data["BERT_Context"].tolist(),
    start_positions=train_data["Start_Positions"].astype(int).tolist(),
    end_positions=train_data["End_Positions"].astype(int).tolist(),
)

val_dataset = QADataset(
    questions=val_data["Cleaned_Title"].tolist(),
    contexts=val_data["BERT_Context"].tolist(),
    start_positions=val_data["Start_Positions"].astype(int).tolist(),
    end_positions=val_data["End_Positions"].astype(int).tolist(),
)

# Modify the DataLoader instantiation with the custom collate function
batch_size = 5
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=True
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, collate_fn=custom_collate, shuffle=False
)
```

#### Model Training
```python
# Setup the optimizer
optimizer = AdamW(model.parameters(), lr=0.9)

# Fine-tune the model
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)

# Initialize the progress bar
progress_bar = tqdm(total=num_epochs * len(train_dataloader), desc="Training Progress")

# Gradient accumulation steps
gradient_accumulation_steps = 4  # Adjust if necessary

best_val_loss = float('inf')
patience = 2  # Number of epochs with no improvement to wait
no_improve_epoch = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Use autocast to enable mixed precision training
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
            if loss is not None:
                loss = loss / gradient_accumulation_steps  # Normalize the loss
        
        # Scale the loss using GradScaler and call backward
        scaler.scale(loss).backward()
        total_loss += scaler.scale(loss).item()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Perform a step using GradScaler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            progress_bar.update(gradient_accumulation_steps)
            logging.info(f"Processed batch {total_loss} in epoch {epoch + 1}")

    logging.info(f"Epoch {epoch+1}, Training Loss: {total_loss/len(train_dataloader)}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()

    logging.info(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_dataloader)}")

progress_bar.close()
```

#### Model Saving
```python
model.save_pretrained("/content/drive/My Drive/Dataspeak/fine_tuned_bert")
tokenizer.save_pretrained("/content/drive/My Drive/Dataspeak/fine_tuned_bert")
```

#### Execution
To execute the notebook, upload it to Google Colab, and run the cells in sequence. Ensure that your Google Drive is mounted correctly and that the paths to the datasets and the save location for the model are correctly specified.
