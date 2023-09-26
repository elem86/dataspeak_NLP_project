import pandas as pd
import re
import spacy
from multiprocessing import Pool
from tqdm import tqdm

# Load Spacy model for lemmatization and stop words
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# Define a function to clean text
def clean_text(text):
    text = re.sub(r"http[s]?://\S+", "", text)
    text = re.sub(r"<.*?>", "", text)  # Modified regex to remove all HTML tags
    text = re.sub(r"\r\n|\n|\t", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(words)


# Define a function to apply cleaning in parallel using multiprocessing
def parallel_clean(df, column_name):
    with Pool() as pool:
        return list(tqdm(pool.imap(clean_text, df[column_name]), total=df.shape[0]))


def main():
    # Load the datasets into Pandas dataframes with 'ISO-8859-1' encoding and error_bad_lines=False
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
    questions_tags_merged = questions.merge(
        tags, left_on="Id", right_on="Id", how="left"
    )
    qa_merged = questions_tags_merged.merge(
        answers,
        left_on="Id",
        right_on="ParentId",
        how="left",
        suffixes=("_question", "_answer"),
    )

    # Drop unnecessary columns and rows
    qa_merged = qa_merged.dropna(
        subset=[
            "Cleaned_Body_question",
            "Cleaned_Title",
            "Tag",
            "Cleaned_Body_answer",
        ]
    )
    qa_merged = qa_merged.reset_index(drop=True)

    # Create BERT_Context by combining cleaned body of the question and body of the answer
    qa_merged["BERT_Context"] = (
        qa_merged["Cleaned_Body_question"] + " " + qa_merged["Cleaned_Body_answer"]
    )

    # Saving the cleaned datasets
    qa_merged.to_csv("QA_cleaned.csv", index=False)
    print("QA_cleaned.csv saved!")
    answers.to_csv("Answers_cleaned.csv", index=False)
    print("Answers_cleaned.csv saved!")
    questions.to_csv("Questions_cleaned.csv", index=False)
    print("Questions_cleaned.csv saved!")


if __name__ == "__main__":
    main()
