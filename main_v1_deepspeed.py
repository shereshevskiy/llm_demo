"""LLM fine tuning demo.
Вариант с DeepSpeed
Код оптимизирован для Apple M1 Pro 16Gb.
"""

import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Устройство для обучения
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# библиотека tokenizers использует параллелизм, на Мак это не поддерживается
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Подготовка папки для сохранения результатов
output_base_path = "./fine_tuned_model_v1_deepspeed"
output_dir = os.getenv("OUTPUT_DIR", output_base_path)  # Если переменная не задана, используется output_base_path
os.makedirs(output_dir, exist_ok=True)

# Загрузка модели
model_name = "facebook/opt-2.7b"  # взяли модель, для загрузки которой не нужно регаться
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
# Аргументы для тренировки
# Инициализация Accelerate с настройками без распределённых вычислений
accelerator = Accelerator(cpu=True, deepspeed_plugin=None)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    per_device_train_batch_size=1,  # Маленький батч из-за ограниченной памяти
    logging_dir="./logs_v1",
    logging_steps=50,
    num_train_epochs=3,
    deepspeed="ds_config.json",  # Указываем DeepSpeed конфигурацию
    fp16=False,  # FP16 отключён для совместимости с MPS
    bf16=False,  # BF16 отключён, так как не поддерживается MPS
)


# # Инициализация Accelerate с настройками
# accelerator = Accelerator(cpu=True, deepspeed_plugin=None)
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     evaluation_strategy="steps",
#     save_steps=100,
#     save_total_limit=1,
#     per_device_train_batch_size=1,  # Уменьшите размер батча
#     logging_dir="./logs_v1",
#     logging_steps=50,
#     num_train_epochs=3,
#     deepspeed="ds_config.json",  # Указываем DeepSpeed конфигурацию
#     fp16=False,  # Убираем fp16, так как MPS его не поддерживает
#     bf16=False,
# )

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Сохраняем результаты
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
