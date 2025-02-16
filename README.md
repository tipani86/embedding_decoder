# Embedding Decoder

An interactive application that attempts to decode and reconstruct text from embeddings through an iterative refinement process. The system uses OpenAI's embedding models and GPT-4 variants to generate and refine text candidates that match the semantic meaning of your input text.

## Features

- Real-time visualization of error convergence
- Dynamic candidate generation with temperature control
- Abstract description generation for semantic guidance
- Comprehensive logging of attempts and results
- Side-by-side comparison of target and best candidate
- Support for multiple embedding and LLM models

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### Available Settings

#### Model Selection
- Embedding Models:
  - text-embedding-3-small
  - text-embedding-3-large
  - text-embedding-ada-002
- LLM Models:
  - gpt-4o-mini
  - gpt-4o

#### Process Configuration
- Temperature (0.0 - 2.0): Controls generation randomness
- Maximum Rounds (1-20): Number of iteration cycles
- Error Threshold (0.0 - 1.0): Target similarity score
- Candidates per Round (1-10): Number of attempts per iteration

### How it Works

1. **Initialization**
   - Convert input text to embedding vector
   - Generate abstract description for semantic guidance
   - Initialize visualization and logging systems

2. **Iterative Process**
   - Each round:
     - Generate candidates based on historical performance
     - Convert candidates to embeddings
     - Calculate similarity scores
     - Update visualization and logs
     - Adapt generation strategy based on error trends

3. **Completion Conditions**
   - Error below threshold achieved
   - Maximum rounds reached
   - Manual stop by user

4. **Results Display**
   - Real-time error convergence chart
   - Detailed log table of attempts
   - Final comparison of target and best candidate
   - Abstract description of target text

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages (see requirements.txt):
  - streamlit >= 1.32.0
  - openai >= 1.12.0
  - numpy >= 1.24.0
  - python-dotenv >= 1.0.0
  - loguru >= 0.7.2
  - plotly >= 5.18.0
