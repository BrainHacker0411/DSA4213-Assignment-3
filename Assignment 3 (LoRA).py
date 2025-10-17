# ============================================================
# Beyoncé & Kanye West QA Fine-Tuning with LoRA (PEFT)
# ============================================================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
from tqdm.auto import tqdm
import time

# -------------------------------------------------------------------
# 1. Load and Filter Dataset
# -------------------------------------------------------------------
print(" Loading dataset...")
dataset = load_dataset("rajpurkar/squad")

def filter_beyonce_kanye(example):
    return example["title"] in ["Beyoncé", "Kanye_West"]

print("🎤 Filtering Beyoncé & Kanye West examples...")
dataset = dataset.filter(filter_beyonce_kanye)
print(f" Filtered dataset: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")

# Combine train + validation, then resplit to ensure both have data
combined = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = combined["train"]
val_data = combined["test"]
print(f" Final Split — Train: {len(train_data)}, Validation: {len(val_data)}")

# -------------------------------------------------------------------
# 2. Load Model and Tokenizer
# -------------------------------------------------------------------
model_name = "distilbert-base-uncased-distilled-squad"
print(f" Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# -------------------------------------------------------------------
# 3. Apply LoRA (Parameter Efficient Fine-Tuning)
# -------------------------------------------------------------------
peft_config = LoraConfig(
    task_type=TaskType.QUESTION_ANS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # DistilBERT attention layers
)
model = get_peft_model(model, peft_config)

# -------------------------------------------------------------------
# 4. Preprocess Dataset for QA
# -------------------------------------------------------------------
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offset_mapping in enumerate(inputs["offset_mapping"]):
        answer = examples["answers"][i]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find context span
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        start_token = context_end
        end_token = context_end
        for j in range(context_start, context_end + 1):
            if offset_mapping[j][0] <= start_char < offset_mapping[j][1]:
                start_token = j
            if offset_mapping[j][0] < end_char <= offset_mapping[j][1]:
                end_token = j
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs.pop("offset_mapping")
    return inputs

print(" Tokenizing and processing dataset...")
train_dataset = train_data.map(preprocess_function, batched=True)
eval_dataset = val_data.map(preprocess_function, batched=True)

# -------------------------------------------------------------------
# 5. Training Configuration
# -------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./beyonce_kanye_peft_qa",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
)

# -------------------------------------------------------------------
# 6. Custom Trainer with tqdm progress
# -------------------------------------------------------------------
class ProgressTrainer(Trainer):
    def train(self, *args, **kwargs):
        total_steps = int(
            len(self.train_dataset)
            / self.args.per_device_train_batch_size
            * self.args.num_train_epochs
        )

        print(f" Starting training ({self.args.num_train_epochs} epochs)...\n")
        start_time = time.time()

        with tqdm(total=total_steps, desc="Training progress", ncols=100) as pbar:
            output = super().train(*args, **kwargs)
            pbar.n = total_steps
            pbar.refresh()

        total_time = time.time() - start_time
        print(f"\n Training finished in {total_time/60:.2f} minutes total")
        return output

# -------------------------------------------------------------------
# 7. Train Model
# -------------------------------------------------------------------
trainer = ProgressTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

trainer.train()

# -------------------------------------------------------------------
# 8. Save Model
# -------------------------------------------------------------------
model.save_pretrained("./beyonce_kanye_peft_qa_model")
print(" Training complete! Model saved to './beyonce_kanye_peft_qa_model'")

# -------------------------------------------------------------------
# 9. Testing the Model
# -------------------------------------------------------------------
from transformers import pipeline

# Load your fine-tuned model
qa_pipeline = pipeline(
    "question-answering",
    model="./beyonce_kanye_peft_qa_model",
    tokenizer="distilbert-base-uncased"
)

# Try a Beyoncé example
context_beyonce = """
Beyoncé released her visual album Lemonade in 2016,
which explored themes of infidelity, forgiveness, and empowerment.
"""
question_beyonce = "What year did Beyoncé release Lemonade?"

result_beyonce = qa_pipeline(question=question_beyonce, context=context_beyonce)
print("Beyoncé result:", result_beyonce)

# Try a Kanye example
context_kanye = """
Kanye West released his album The Life of Pablo in 2016.
The album featured songs such as Famous, Father Stretch My Hands, and Ultralight Beam.
"""
question_kanye = "Which album did Kanye West release in 2016?"

result_kanye = qa_pipeline(question=question_kanye, context=context_kanye)
print("Kanye result:", result_kanye)

# -------------------------------------------------------------------
# 10. Evaluate Model Performance (F1 & Exact Match)
# -------------------------------------------------------------------
import evaluate

print("\n📏 Evaluating model performance on validation set...")

# Load the official SQuAD metric
squad_metric = evaluate.load("squad")

# Define QA pipeline for inference
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

predictions = []
references = []

for example in tqdm(eval_dataset, desc="Evaluating", ncols=100):
    context = example["context"]
    question = example["question"]
    true_answers = example["answers"]

    try:
        result = qa_pipeline(question=question, context=context)
        pred_text = result["answer"]
    except Exception:
        pred_text = ""

    predictions.append({
        "id": example["id"],
        "prediction_text": pred_text
    })
    references.append({
        "id": example["id"],
        "answers": true_answers
    })

results = squad_metric.compute(predictions=predictions, references=references)

print("\n Evaluation Results:")
print(f"Exact Match (EM): {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")

# -------------------------------------------------------------------
# 11. Evaluate Model Performance (F1 & Exact Match)
# -------------------------------------------------------------------
from tqdm import tqdm
import evaluate
import numpy as np
import pandas as pd

# Load the SQuAD metric from `evaluate`
metric = evaluate.load("squad")

predictions = []
references = []

for example in tqdm(eval_dataset):
    question = example["question"]
    context = example["context"]
    true_answers = example["answers"]["text"]

    pred = qa_pipeline(question=question, context=context)
    pred_text = pred["answer"].strip()

    predictions.append(pred_text)
    references.append(true_answers[0].strip() if len(true_answers) > 0 else "")

# Compute per-example F1 and EM
results = []
for i, (pred, ref) in enumerate(zip(predictions, references)):
    scores = metric.compute(
        predictions=[{"id": str(i), "prediction_text": pred}],
        references=[{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}}],
    )
    results.append({
        "index": i,
        "question": eval_dataset[i]["question"],
        "context": eval_dataset[i]["context"],
        "true_answer": ref,
        "predicted_answer": pred,
        "EM": scores["exact_match"],
        "F1": scores["f1"],
    })

# Convert to DataFrame and sort by F1
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="F1", ascending=True)

# Display the 10 worst predictions
for _, row in df_sorted.head(10).iterrows():
    print(f" Question: {row['question']}")
    print(f" Context: {row['context'][:250]}...")
    print(f" True Answer: {row['true_answer']}")
    print(f" Predicted: {row['predicted_answer']}")
    print(f"F1: {row['F1']:.2f}, EM: {row['EM']:.2f}")
    print("—" * 80)
# -------------------------------------------------------------------
# 12. Just me messing around with the model
# -------------------------------------------------------------------
from transformers import pipeline

# Load your fine-tuned model
qa_pipeline = pipeline(
    "question-answering",
    model="./beyonce_kanye_peft_qa_model",
    tokenizer="distilbert-base-uncased-distilled-squad"
)

# Example context and question
context = "Beyonce married Jay-Z in 2008"
question = "Who did Beyonce marry?"

result = qa_pipeline(question=question, context=context)
print("Answer:", result['answer'])
print("Confidence:", result['score'])

#-----------------------------------------------------------------
#Further Analysis
#-----------------------------------------------------------------

def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    ratio = (trainable / total) * 100
    print(f"\n Model Parameter Summary (LoRA):")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Total parameters:     {total:,}")
    print(f"   Trainable ratio:      {ratio:.2f}%")

print_trainable_parameters(model)

# -------------------------------------------------------------------
# 13. Segment-Based Evaluation: Performance by Question Type
# -------------------------------------------------------------------
from collections import defaultdict
import re

print("\n Segmenting evaluation results by question type...")

# Function to detect question type (simple heuristic)
def get_question_type(question):
    match = re.match(r"^(what|who|when|where|why|how|which|whom)\b", question.lower())
    return match.group(1) if match else "other"

# Prepare containers for grouped scores
grouped_scores = defaultdict(lambda: {"f1": [], "em": []})

# Evaluate per example again (using earlier predictions/references)
for i, (pred, ref) in enumerate(zip(predictions, references)):
    qtype = get_question_type(eval_dataset[i]["question"])
    scores = metric.compute(
        predictions=[{"id": str(i), "prediction_text": pred}],
        references=[{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}}],
    )
    grouped_scores[qtype]["f1"].append(scores["f1"])
    grouped_scores[qtype]["em"].append(scores["exact_match"])

# Compute average per question type
summary = []
for qtype, scores in grouped_scores.items():
    avg_f1 = sum(scores["f1"]) / len(scores["f1"])
    avg_em = sum(scores["em"]) / len(scores["em"])
    summary.append({"Type": qtype.capitalize(), "Avg F1": avg_f1, "Avg EM": avg_em, "Count": len(scores["f1"])})

# Convert to DataFrame
df_summary = pd.DataFrame(summary).sort_values(by="Avg F1", ascending=False)

print("\n Performance by Question Type (LoRA):")
print(df_summary.to_string(index=False, formatters={
    "Avg F1": "{:.2f}".format,
    "Avg EM": "{:.2f}".format,

}))
