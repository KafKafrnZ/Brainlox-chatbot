
# Brainlox Chatbot API

Welcome to the Brainlox Chatbot API! This application is designed to provide an AI-powered chatbot capable of answering questions and retrieving information from web-based resources.

## Features
- **Web Data Integration**: Loads and processes content from specified URLs.
- **AI-Powered Chatbot**: Utilizes cutting-edge language models for intelligent responses.
- **Conversation Memory**: Keeps track of chat history for better context.
- **Rate Limiting**: Ensures fair usage by limiting API requests.
- **RESTful API**: Simple and user-friendly endpoints for easy integration.

---

## How It Works
1. **Load Data**: The chatbot fetches data from the web, splits it into manageable chunks, and stores it for retrieval.
2. **AI Model**: Uses Hugging Face models for language understanding and response generation.
3. **Query Handling**: Processes user queries, retrieves relevant information, and returns answers with sources.

---

## Getting Started

### Requirements
- **Python**: Version 3.8 or later
- **Dependencies**: Install the required libraries by running:
  ```bash
  pip install -r requirements.txt
  ```
- **Environment Variables**: Set up a `.env` file with the following keys:
  ```plaintext
  HF_API_TOKEN=your_huggingface_api_token
  ```
  > **Note**: You need a **read Hugging Face API token** to enable vector embedding and use the AI models.

### Running the App
1. Clone the repository and navigate to the project directory.
2. Start the Flask server:
   ```bash
   python chatbot.py
   ```
3. Visit `http://localhost:5000` in your browser to access the API.

---

## Endpoints

### Home
- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a welcome message.

### Chat UI
- **URL**: `/ui`
- **Method**: `GET`
- **Description**: Serves a simple HTML-based chat interface.

### Chatbot API
- **URL**: `/chat`
- **Method**: `POST`
- **Request**: JSON body with a `query` field.
  ```json
  {
    "query": "What is AI?"
  }
  ```
- **Response**: JSON object containing the chatbot's response and source snippets.
  ```json
  {
    "response": "AI stands for Artificial Intelligence...",
    "sources": ["Source snippet 1", "Source snippet 2"]
  }
  ```

---

## Key Configuration

- **HF_API_TOKEN**: Your Hugging Face API token for accessing AI models and generating vector embeddings.
- **WEB_URLS**: List of URLs the chatbot pulls data from.
- **Rate Limits**: Default limits are 200 requests per day and 50 per hour.

---

## Troubleshooting

- **No Token Found**: Ensure the `.env` file contains your Hugging Face API token.
- **No Data Loaded**: Check that the URLs in `WEB_URLS` are accessible and contain valid data.
- **Server Errors**: Review the logs for any exceptions or configuration issues.

---

## Contributing
We welcome contributions! Feel free to submit pull requests or issues for improvements and bug fixes.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
