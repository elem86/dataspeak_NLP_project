import pandas as pd
import re
import string
import spacy
from multiprocessing import Pool
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean_text(text):
    # Encode <code> and </code> tags
    text = re.sub(r"<code>", "[CODE]", text)
    text = re.sub(r"</code>", "[/CODE]", text)

    # Optional: Encode URLs
    text = re.sub(r"(http[s]?://\S+)", "[URL]", text)

    # Remove other HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Replace newline and tab characters with spaces
    text = re.sub(r"\r\n|\n|\t", " ", text)

    # Keep punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s" + string.punctuation + "]", "", text)

    return text


def parallel_clean(df, column_name):
    with Pool() as pool:
        return list(tqdm(pool.imap(clean_text, df[column_name]), total=df.shape[0]))


def main():
    answers = pd.read_csv("Answers.csv", encoding="ISO-8859-1")
    questions = pd.read_csv("Questions.csv", encoding="ISO-8859-1")

    # Preprocessing steps
    def preprocess(df):
        df = df.dropna()
        df = df.drop_duplicates()
        return df

    answers = preprocess(answers)
    questions = preprocess(questions)

    # Applying cleaning 'Body' and 'Title' columns of questions and answers datasets in parallel
    questions["Cleaned_Body"] = parallel_clean(questions, "Body")
    answers["Cleaned_Body"] = parallel_clean(answers, "Body")
    questions["Cleaned_Title"] = parallel_clean(questions, "Title")

    # Drop unnecessary columns
    answers = answers.drop("Body", axis=1)
    questions = questions.drop(["Body", "Title"], axis=1)

    # Merging DataFrames
    qa_merged = questions.merge(
        answers,
        left_on="Id",
        right_on="ParentId",
        how="left",
        suffixes=("_question", "_answer"),
    )

    # Filter out answers with Score_answer 0 and below
    qa_merged = qa_merged[
        qa_merged.get("Score_answer", 1) > 0
    ]  # Assuming Score_answer is the column name for the answer score

    # Merge title and body for questions
    qa_merged["Title_Body_Q"] = (
        qa_merged["Cleaned_Title"] + " \n\n " + qa_merged["Cleaned_Body_question"]
    )

    # Drop unnecessary columns
    qa_merged = qa_merged.reset_index(drop=True)

    # Create Context by combining merged title and body of the question and body of the answer
    qa_merged["Context"] = (
        qa_merged["Title_Body_Q"] + " " + qa_merged["Cleaned_Body_answer"]
    )

    # Keep only the specified columns
    columns_to_keep = [
        "Id_question",
        "Title_Body_Q",
        "Id_answer",
        "ParentId",
        "Cleaned_Body_answer",
        "Context",
    ]
    qa_merged = qa_merged[columns_to_keep]

    # Saving the cleaned datasets
    qa_merged.to_csv("QA_ready.csv", index=False)
    print("QA_ready.csv saved!")


if __name__ == "__main__":
    main()
