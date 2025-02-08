# EcoEats Server

This is the remote server that performs the actual image analysis behind our app.

## Getting Started

### 1. Initialize a Conda Environment  
Run the following command to create and activate a Conda environment:  

```sh
conda create --name ecoeats-env python=3.9 -y
conda activate ecoeats-env
```

### 2. Install Dependencies  
Ensure you have all required dependencies by installing them from `requirements.txt`:  

```sh
pip install -r requirements.txt
```

### 3. Install Ollama  
Follow the instructions to install Ollama from [Ollama's official site](https://ollama.ai).
```sh
curl -fsSL https://ollama.ai/install.sh | sh
```

### 4. Run Ollama  
Start Ollama using the following command:  
```sh
ollama run ollama
```

### 5. Start the Server  
Run the server script to begin processing requests:  
```sh
python src/server.py
```

## Usage
After starting the server, it will be ready to accept image analysis requests via API calls.


## License
This project is licensed under the [MIT License](LICENSE).
