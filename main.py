"""LLM fine tuning demo."""

import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Устройство для обучения
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Загрузка модели
model_name = "facebook/opt-2.7b"
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
# Загружаем данные
data = load_dataset("json", data_files="data/data.jsonl")["train"]

# Разделяем данные на тренировочные и тестовые
train_test_split = data.train_test_split(test_size=0.2)

# Функция для обработки данных
def preprocess_data(examples):
    # Создаём input_text из instruction и input
    inputs = [
        f"Instruction: {instruction} Input: {input_text}" 
        for instruction, input_text in zip(examples["instruction"], examples["input"])
    ]

    # Токенизация input_text
    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)

    # Токенизация labels из output
    with tokenizer.as_target_tokenizer():
        tokenized_labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)

    # Возвращаем токенизированные данные
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_labels["input_ids"]
    }

# Применяем обработку
train_dataset = train_test_split["train"].map(preprocess_data, batched=True)
eval_dataset = train_test_split["test"].map(preprocess_data, batched=True)


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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Получаем путь из переменной окружения
output_dir = os.getenv("OUTPUT_DIR", "./fine_tuned_model")  # Если переменная не задана, используется "./fine_tuned_model"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
