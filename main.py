"""LLM fine tuning demo."""

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Устройство для обучения
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Загрузка модели
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,  # Уменьшаем потребление памяти
)
model = model.to(device)

# Подключение LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Подготовка данных
data = load_dataset("json", data_files="data.jsonl")

def preprocess_data(examples):
    return {
        "input_text": [
            f"Instruction: {ex['instruction']} Input: {ex['input']}" for ex in examples
        ],
        "labels": [ex["output"] for ex in examples],
    }

tokenized_data = data.map(preprocess_data, batched=True)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=1,  # Маленький батч
    gradient_accumulation_steps=8,  # Накопление градиентов
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=False,  # Убираем fp16, так как MPS его не поддерживает
    bf16=False,
    report_to="none",
)

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
