# News Analyser

A Streamlit web application built as a minor project during the 3rd year of engineering. It lets users input up to 3 news article URLs, processes their content using LangChain and OpenAI, and answers questions about the articles with cited sources.

## Stack

- Streamlit
- LangChain
- FAISS
- OpenAI GPT and Embeddings

## Setup

1. Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/yourusername/News_Analyser.git
    cd News_Analyser
    pip install -r Requirements.txt
    ```

2. Copy `.env.example` to `.env` and fill in your OpenAI API key:

    ```bash
    cp .env.example .env
    ```

    `.env`:
    ```
    OPENAI_API_KEY=your-openai-api-key-here
    ```

3. Run the app:

    ```bash
    streamlit run main.py
    ```

## Usage

- Paste up to 3 news article URLs in the sidebar.
- Click "Analyze Sources" to process and index the content.
- Type a question in the main input field to get an answer with sources.

    ```

## Usage

1. **Run the app**:

    After setting up the environment, run the app locally:

    ```bash
    streamlit run main.py
    ```

2. **Interact with the App**:

    - Open the app in your browser at `http://localhost:8501`.
    - Paste up to 3 news URLs into the sidebar's input fields.
    - Click **"Analyze"** to process the articles.
    - Once the analysis is complete, ask a question in the main input field, and the app will generate an answer along with relevant sources.

## Project Structure

```bash
.
├── app.py              # Main application file
├── ApiKey.py           # Contains OpenAI API key
├── requirements.txt    # Python dependencies
├── faiss_index         # Directory where FAISS index is saved (auto-created)
└── README.md           # Project documentation
