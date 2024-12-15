# ****LLM Fine-Tuning Demo****

This project demonstrates the process of fine-tuning large language models (LLMs) on Macs with M1/M2 processors, including the use of PyTorch with MPS (Metal Performance Shaders) support.

## ****Project Description****

The project is designed to showcase the fine-tuning of open-source models, such as LLaMA or Mistral. It includes environment setup, data loading, the training process using the LoRA method, and testing of the results.

The project also features an optimized version of the code using the DeepSpeed library.

Note: On Macs with M1 processors, the DeepSpeed-optimized code cannot run due to the lack of support for some critical DeepSpeed commands on Mac.

---

## **Installation**

To replicate the project, follow these steps:

### **1. Clone the repository**

```bash
git clone https://github.com/shereshevskiy/llm_demo.git
cd llm_demo
```

### **2. Create a virtual environment**

```
python -m venv llm_env
source llm_env/bin/activate
```

**Note:** Use Python >= 3.10 (e.g., 3.10.12).

**3. Install PyTorch**

Install PyTorch:

```
./install_pytorch.sh
```

**Note:**

* Ensure that the script **install_pytorch.sh** has execution permissions.
* The script is optimized for use on Apple M1/M2 processors.

**Alternatively, you can run:**

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### **4. Install other dependencies**

```
pip install -r requirements.txt
```

**Environment Check**

After installation, run the following command to ensure PyTorch is properly set up:

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

If the output is  **True** , PyTorch supports MPS acceleration on your device.

## **Running the Project**

### **Setup**

If the project requires configuration, such as editing configuration files (e.g., based on a **.env.example** template), please do so.

This demo project does not currently require configuration.

### **Starting Training**

To start the main training script, run:

```
python main.py
```

The training results will be saved in the **fine_tuned_model/** folder.

### **Project Structure**

llm_demo/   
```   
├── README.md   # Project description   
├── docker-compose.yml # Docker Compose configuration   
├── dockerfile # Dockerfile for building the image   
├── fine_tuned_model/  # Folder for saving results (mounted to the host)   
├── requirements.txt # Dependencies (excluding PyTorch)   
├── install_pytorch.sh # PyTorch installation script   
├── main.py  # Main code   
├── .env.example # Example configuration file   
├── data/  # Project data   
├── fine_tuned_model/  # Output folder for the model   
```

### **Example Data**

Training data is stored in the **data** folder in JSONL format. Example file:

```
{"instruction": "Translate to French", "input": "Hello, world!", "output": "Bonjour, le monde!"}
{"instruction": "Summarize", "input": "AI is transforming industries.", "output": "AI revolutionizes industries."}
```

### **Dependencies**

#### **Installing PyTorch**

PyTorch is installed via a separate script **install_pytorch.sh** with an optimized index or using the following command (this must be done **before** installing dependencies from **requirements.txt**):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### **Main Dependencies**

```
transformers>=4.33.0
datasets>=2.14.0
peft>=0.4.0
accelerate>=0.21.0
```

## **Running with Docker Compose**

### **1. Ensure Docker and Docker Compose are installed**

If Docker and Docker Compose are not installed, follow the instructions for your operating system:

* [Install Docker](https://docs.docker.com/get-docker/)

* [Install Docker Compose](https://docs.docker.com/compose/install/)

### **2. Build and run the container**

From the project root, run the following command:

```
docker compose up --build
```

This command:

 **Builds the Docker image based on thedockerfile .**

    Runs the container with folder mounting for saving results.

### **3. Save results**

The results (model, logs, etc.) will be saved in the **./fine_tuned_model** folder on your host machine. This folder is mounted in the container at  **/app/fine_tuned_model** .

### **4. View logs**

Application logs will be displayed in the terminal. To stop the application, press  **Ctrl+C** .

### **5. Stop the container**

To stop and remove the container, run:

```
docker compose down
```

### **6. Check results**

After the container finishes, check the **fine_tuned_model** folder in the project root. You should find:

* Saved model files.
* Application logs (if enabled).

## **Contact**

If you have any questions or suggestions, feel free to reach out at [d.shereshevskiy@gmail.com](mailto:d.shereshevskiy@gmail.com).
