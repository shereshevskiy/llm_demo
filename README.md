# LLM Fine-Tuning Demo

Этот проект демонстрирует процесс файн-тюнинга больших языковых моделей (LLM) на Mac с процессорами M1/M2, включая использование PyTorch с поддержкой MPS (Metal Performance Shaders).

## Описание проекта

Проект предназначен для демонстрации файн-тюнинга open-source моделей, таких как LLaMA или Mistral. Он включает настройку среды, загрузку данных, процесс обучения с использованием метода LoRA, и тестирование результатов.

---

## Установка

Для воспроизведения проекта выполните следующие шаги:

### 1. Клонируйте репозиторий

```
git clone https://github.com/shereshevskiy/llm_demo.git
cd llm_demo
```

### **2. Создайте виртуальное окружение**

```
python -m venv llm_env
```

```
source llm_env/bin/activate
```

**Примечание:** используйте Python >=3.10 (здесь 3.10.12)

### **3. Установите PyTorch**

Установите PyTorch

```
./install_pytorch.sh
```

**Примечание** :

1. Убедитесь, что скрипт **install_pytorch.sh** имеет права на выполнение,
2. Здесь оптимизированно для работы на процессорах Apple M1/M2,

**Вместо можно выполнить**

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **4. Установите остальные зависимости**

```
pip install -r requirements.txt
```

**Проверка среды**

После установки выполните следующую команду, чтобы убедиться, что PyTorch настроен правильно:

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

Если вывод **True**, PyTorch поддерживает ускорение через MPS на вашем устройстве.

## **Запуск проекта**

### **Настройка**

Если проект требует конфигурации, например настройку конфигурационных файлов (файла на базе шаблона .env.example и тп) - сделайте это.

В данном демо проекте пока этого не требуется.

### **Запуск обучения**

Для запуска основного скрипта обучения выполните:

```
python main.py
```

Результаты обучения будут сохранены в папке **fine_tuned_model/**.

### **Структура проекта**

llm_demo/
├── README.md             # Описание проекта
├── requirements.txt      # Зависимости (кроме PyTorch)
├── install_pytorch.sh    # Установка PyTorch
├── main.py               # Основной код
├── .env.example          # Пример конфигурационного файла
├── data/                 # Данные проекта
├── fine_tuned_model/     # Выходная папка для модели

### **Пример данных**

Данные для обучения хранятся в папке **data** в формате JSONL. Пример файла:

```
{"instruction": "Translate to French", "input": "Hello, world!", "output": "Bonjour, le monde!"}{"instruction": "Summarize", "input": "AI is transforming industries.", "output": "AI revolutionizes industries."}
```

### **Зависимости**

#### **Установка PyTorch**

PyTorch устанавливается через отдельный скрипт **install_pytorch.sh** с указанием оптимизированного индекса или с помощью кода (сделайте это **до** установки зависимостей из.txt):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **Основные зависимости**

```
transformers>=4.33.0
datasets>=2.14.0
peft>=0.4.0
accelerate>=0.21.0
```

### **Контакт**

Если у вас возникли вопросы или предложения, пишите на [d.shereshevskiy@gmail.com](mailto:d.shereshevskiy@gmail.com).
