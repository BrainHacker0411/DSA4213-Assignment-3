# ============================================================
# Beyoncé & Kanye West QA Fine-Tuning with IA³ (PEFT)
# ============================================================

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import get_peft_model, IA3Config, TaskType
import torch
from tqdm.auto import tqdm
import time

# -------------------------------------------------------------------
# 1. Load and Filter Dataset
# -------------------------------------------------------------------
print("Loading dataset...")
dataset = load_dataset("rajpurkar/squad")

def filter_beyonce_kanye(example):
    return example["title"] in ["Beyoncé", "Kanye_West"]

print(" Filtering Beyoncé & Kanye West examples...")
dataset = dataset.filter(filter_beyonce_kanye)
print(f" Filtered dataset: Train={len(dataset['train'])}, Val={len(dataset['validation'])}")

# Combine and resplit
combined = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_data = combined["train"]
val_data = combined["test"]
print(f" Final Split — Train: {len(train_data)}, Validation: {len(val_data)}")

# -------------------------------------------------------------------
# 2. Load Model and Tokenizer
# -------------------------------------------------------------------
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
print(f" Loaded model: {model_name}")

# -------------------------------------------------------------------
# 3. Apply IA³ (PEFT)
# -------------------------------------------------------------------
peft_config = IA3Config(
    task_type=TaskType.QUESTION_ANS,
    target_modules=["q_lin", "v_lin", "output_layer"],
    feedforward_modules=["output_layer"]
)
model = get_peft_model(model, peft_config)

# -------------------------------------------------------------------
# 4. Preprocess Dataset
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

    start_positions, end_positions = [], []
    for i, offset_mapping in enumerate(inputs["offset_mapping"]):
        answer = examples["answers"][i]
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Identify the context tokens
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
    output_dir="./beyonce_kanye_ia3_qa",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    logging_dir="./logs_ia3",
    logging_steps=50,
    report_to="none",
)

# -------------------------------------------------------------------
# 6. Train Model
# -------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

trainer.train()

# -------------------------------------------------------------------
# 7. Save Model
# -------------------------------------------------------------------
model.save_pretrained("./beyonce_kanye_ia3_model")
print(" Training complete! Model saved to './beyonce_kanye_ia3_model'")

# -------------------------------------------------------------------
# 8. Evaluate IA³ Model — EM, F1, and Error Analysis
# -------------------------------------------------------------------
from transformers import pipeline
from tqdm import tqdm
import evaluate
import pandas as pd

print(" Running evaluation on validation dataset...")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

predictions = []
references = []

for example in tqdm(val_data):
    question = example["question"]
    context = example["context"]
    true_answers = example["answers"]["text"]

    pred = qa_pipeline(question=question, context=context)
    pred_text = pred["answer"].strip()

    predictions.append(pred_text)
    references.append(true_answers[0].strip() if len(true_answers) > 0 else "")

#  Compute EM & F1 using the official SQuAD metric
metric = evaluate.load("squad")
formatted_predictions = [{"id": str(i), "prediction_text": p} for i, p in enumerate(predictions)]
formatted_references = [{"id": str(i), "answers": {"text": [r], "answer_start": [0]}} for i, r in enumerate(references)]
results = metric.compute(predictions=formatted_predictions, references=formatted_references)

print("\n Evaluation Results:")
print(f"Exact Match (EM): {results['exact_match']:.2f}")
print(f"F1 Score: {results['f1']:.2f}")

# -------------------------------------------------------------------
# 9. Inspect Worst-Performing Predictions
# -------------------------------------------------------------------
print("\n🔍 Analyzing lowest F1 examples...")
detailed_results = []
for i, (pred, ref) in enumerate(zip(predictions, references)):
    score = metric.compute(
        predictions=[{"id": str(i), "prediction_text": pred}],
        references=[{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}}],
    )
    detailed_results.append({
        "index": i,
        "question": val_data[i]["question"],
        "context": val_data[i]["context"],
        "true_answer": ref,
        "predicted_answer": pred,
        "EM": score["exact_match"],
        "F1": score["f1"]
    })

df = pd.DataFrame(detailed_results)
df_sorted = df.sort_values(by="F1", ascending=True)

for _, row in df_sorted.head(5).iterrows():
    print(f"\n Question: {row['question']}")
    print(f" Context: {row['context'][:500]}...")
    print(f" True Answer: {row['true_answer']}")
    print(f" Predicted: {row['predicted_answer']}")
    print(f"F1: {row['F1']:.2f}, EM: {row['EM']:.2f}")
    print("—" * 80)

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
    print(f"\n Model Parameter Summary (IA³):")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Total parameters:     {total:,}")
    print(f"   Trainable ratio:      {ratio:.2f}%")

print_trainable_parameters(model)

# -------------------------------------------------------------------
# 10. Segment-Based Evaluation: Performance by Question Type
# -------------------------------------------------------------------
from collections import defaultdict
import re

print("\n Segmenting IA³ evaluation results by question type...")

# Simple function to identify question type
def get_question_type(question):
    match = re.match(r"^(what|who|when|where|why|how|which|whom)\b", question.lower())
    return match.group(1) if match else "other"

# Prepare grouped scores
grouped_scores = defaultdict(lambda: {"f1": [], "em": []})

# Evaluate per question type
for i, (pred, ref) in enumerate(zip(predictions, references)):
    qtype = get_question_type(val_data[i]["question"])
    scores = metric.compute(
        predictions=[{"id": str(i), "prediction_text": pred}],
        references=[{"id": str(i), "answers": {"text": [ref], "answer_start": [0]}}],
    )
    grouped_scores[qtype]["f1"].append(scores["f1"])
    grouped_scores[qtype]["em"].append(scores["exact_match"])

# Compute average scores per question type
summary = []
for qtype, scores in grouped_scores.items():
    avg_f1 = sum(scores["f1"]) / len(scores["f1"])
    avg_em = sum(scores["em"]) / len(scores["em"])
    summary.append({
        "Type": qtype.capitalize(),
        "Avg F1": avg_f1,
        "Avg EM": avg_em,
        "Count": len(scores["f1"])
    })

# Convert to DataFrame and display nicely
df_summary = pd.DataFrame(summary).sort_values(by="Avg F1", ascending=False)

print("\n IA³ Performance by Question Type (IA3):")
print(df_summary.to_string(index=False, formatters={
    "Avg F1": "{:.2f}".format,
    "Avg EM": "{:.2f}".format,

}))
