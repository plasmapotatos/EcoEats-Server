# EcoEats Server

<img src="logo.png" alt="EcoEats Logo" width="200">

This is the remote server that performs the actual image analysis behind our app. For the frontend, [this](https://github.com/plasmapotatos/EcoEats-App) is the link to the app.

## Under Development

This project is still in development. Although the first endpoint (suggesting eco-friendly alternatives to foods) is already implemented we are currently working to add an endpoint which returns a recipe and image for a dish from an inputted ingredient list. We're experimenting with two approaches for generating accurate and eco-conscious responses:
  - Finetuning an open-source LLM on recipe and sustainability data.
  - Implementing **Retrieval-Augmented Generation (RAG)** using a vector store and embedding models
  - **Transparent Source Citation**  
    For credibility and transparency, our system will cite the sources of recipe data or sustainability facts used during generation.

## Getting Started

### 1. Initialize a Conda Environment  
Run the following command to create and activate a Conda environment:  

```sh
conda create --name ecoeats-env python=3.12 -y
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
After starting the server, it will be ready to accept image analysis requests.


## License
This project is licensed under the [MIT License](LICENSE).
