# Semantic Search App

This is a Streamlit application for semantic search using Sentence Transformers and Azure OpenAI.

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Git

### Clone the Repository

```sh
git clone https://github.com/jf2fa/sapp.git
cd sapp

python -m venv env
.\env\Scripts\activate

pip install -r requirements.txt

docker build -t semantic-search-app .
docker run -p 8501:8501 semantic-search-app
