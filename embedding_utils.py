import os
from typing import List, Tuple
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
from loguru import logger

# Configure logger
logger.add(
    "debug.log",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="DEBUG",
    rotation="100 MB"
)

def init_openai_client():
    """Initialize OpenAI client with proper error handling."""
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not found in environment variables")
        raise ValueError(
            "OpenAI API key not found! Please set the OPENAI_API_KEY environment variable. "
            "You can do this by:\n"
            "1. Creating a .env file in the project root\n"
            "2. Adding the line: OPENAI_API_KEY=your_api_key_here\n"
            "3. Replace 'your_api_key_here' with your actual OpenAI API key"
        )
    
    logger.info("OpenAI client initialized successfully")
    return OpenAI(api_key=api_key)

# Initialize the OpenAI client
try:
    client = init_openai_client()
except ValueError as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings for texts using OpenAI's API. Works with single or multiple texts."""
    if isinstance(texts, str):
        texts = [texts]
    
    logger.debug(f"Getting embeddings for {len(texts)} texts using model {model}")
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        logger.debug(f"Successfully got embeddings of dimension {len(response.data[0].embedding)}")
        return [data.embedding for data in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        raise

def calculate_vector_difference(vec1: List[float], vec2: List[float]) -> float:
    """Calculate the cosine distance between two vectors."""
    try:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        similarity = np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))
        distance = 1 - similarity
        logger.debug(f"Calculated vector difference: {distance:.4f}")
        return distance
    except Exception as e:
        logger.error(f"Error calculating vector difference: {e}")
        raise

def get_abstract_description(text: str, llm_model: str = "gpt-4o-mini") -> str:
    """Get an abstract description of the text without revealing actual words."""
    logger.info("Generating abstract description of target text")
    
    system_prompt = """You are an assistant that provides insightful hints about text without revealing its content.
    Your task is to give 3-4 powerful hints that capture the essence and context of the text.
    
    Focus on these aspects (pick the most relevant):
    - Cultural context or references
    - Domain or field of knowledge
    - Emotional or psychological state
    - Perspective or point of view
    - Time period or setting
    - Target audience or context
    - Purpose or intended effect
    - Common situations or scenarios
    - Relationship dynamics
    - Underlying assumptions or beliefs
    
    Each hint should be:
    - Culturally or contextually informative
    - Focused on meaning and intent
    - Helpful for understanding the "world" of the text
    - Free of actual words from the text
    - Free of direct synonyms
    
    Example format:
    1. This reflects [type of perspective] common in [specific context]
    2. Similar to expressions used in [situation/field/time]
    3. Conveys a feeling typical of [scenario/relationship/state]
    
    Make each hint reveal something meaningful about the text's context without giving away specific words."""

    user_prompt = f"""Provide 3-4 insightful hints about this text: "{text}"
    
    Requirements:
    - Focus on cultural, contextual, and thematic aspects
    - Reveal the "world" or context the text exists in
    - Help understand the perspective or situation
    - NO actual words from the text
    - NO direct synonyms
    
    Think about:
    - What kind of person might say this?
    - In what situation would this be said?
    - What cultural context does this reflect?
    - What assumptions or beliefs are behind it?
    
    Format as a numbered list (3-4 points)."""

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        description = response.choices[0].message.content.strip()
        logger.debug(f"Generated abstract description: {description}")
        return description
        
    except Exception as e:
        logger.error(f"Error generating abstract description: {e}")
        raise

def generate_candidates(
    target_text: str,
    previous_results: List[Tuple[str, float, int]] = None,  # Now includes round number
    round_number: int = 1,
    num_candidates: int = 5,
    llm_model: str = "gpt-4o-mini",
    abstract_description: str = None,
    temperature: float = 0.5
) -> List[str]:
    """Generate candidate texts using specified LLM model."""
    logger.info(f"Generating {num_candidates} candidates for round {round_number} using {llm_model} (temp={temperature})")
    
    # Calculate target text word count
    word_count = len(target_text.split())
    logger.debug(f"Target text word count: {word_count}")
    
    if previous_results:
        # Sort all results by error (descending)
        sorted_results = sorted([(text, error) for text, error, _ in previous_results], 
                              key=lambda x: x[1], reverse=True)
        
        # Build history string
        previous_results_str = "PREVIOUS ATTEMPTS (sorted by error, worst to best):\n\n" + "\n".join([
            f"Error: {error:.4f} - Candidate: '{text}'" + 
            (f" [WARNING: {len(text.split())} words instead of {word_count}!]" 
             if len(text.split()) != word_count else "")
            for text, error in sorted_results
        ])
        
        # Calculate best error so far
        best_error = min(error for _, error, _ in previous_results)
        logger.debug(f"Historical results:\n{previous_results_str}")
        logger.debug(f"Best error so far: {best_error:.4f}")
    else:
        previous_results_str = "No previous results available yet"
        logger.debug("No previous results available")

    system_prompt = f"""You are participating in a game where you need to generate text variations that will result in similar embedding vectors to an unknown target text.
    You will receive feedback in the form of error scores (0.0 to 1.0) for each candidate, where lower scores mean the embedding is closer to the target.
    Your goal is to generate candidates that will achieve the lowest possible error score.
    
    CRITICAL REQUIREMENTS:
    1. EXACT Word Count: All candidates MUST be EXACTLY {word_count} words long. No exceptions.
       - Articles (a, an, the) count as words!
       - Count every single word, including all articles
       - No exceptions or special cases
    
    2. Dynamic Variation Strategy:
       - Start with high variation to explore the embedding space broadly
       - As error scores decrease, adjust variation intelligently:
         * High errors (>0.2): Maintain variation to explore different approaches while noting how historical candidates performed to inform your next attempts
         * Low errors (<0.2): Focus more on refining successful patterns while still maintaining some variation
       - Use your judgment based on error trends to determine the right balance

    Target Information:
    - Required word count: EXACTLY {word_count} words (including all articles)
    - Abstract description: {abstract_description or "Not available"}
    - The description is intended to not reveal the actual result, but rather provide hints about how to construct the text. When you have seen enough similar patterns in the historical results, take a deep breath and you can usually make an educated guess.
    
    Output must be a valid JSON object with a 'result' key containing an array of exactly {num_candidates} strings.
    Each string must be EXACTLY {word_count} words long."""

    user_prompt = f"""Round {round_number}

COMPLETE HISTORY OF ALL ROUNDS:
{previous_results_str}

Generate {num_candidates} new candidates that are EXACTLY {word_count} words long.

Strategy requirements:
1. ANALYZE HISTORY:
   - Study the complete history of attempts and their errors
   - Learn from both successful and unsuccessful patterns
   - Notice what approaches led to lower errors
   - Consider both content and structure patterns

2. STRICT WORD COUNT:
   - Each candidate MUST be EXACTLY {word_count} words
   - Articles (a, an, the) COUNT as words!
   - Count EVERY single word, including ALL articles
   - Double-check your count before submitting
   - No punctuation tricks to change word count

3. SMART EXPLORATION:
   - Adjust variation based on error trends
   - Use the abstract description as a guide
   - Avoid exact repetition of previous candidates
   - Balance between exploration and refinement

Output a JSON object with a 'result' key containing an array of {num_candidates} strings.
VERIFY each string is EXACTLY {word_count} words (including articles) before submitting."""

    logger.debug(f"System prompt:\n{system_prompt}")
    logger.debug(f"User prompt:\n{user_prompt}")

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=temperature  # Use the provided temperature
        )
        
        logger.debug(f"Raw API response content:\n{response.choices[0].message.content}")
        
        # Parse the JSON response
        try:
            parsed_response = json.loads(response.choices[0].message.content)
            logger.debug(f"Parsed JSON response:\n{parsed_response}")
            
            if "result" not in parsed_response:
                logger.error(f"'result' key not found in parsed response. Keys found: {list(parsed_response.keys())}")
                raise KeyError("Response JSON does not contain 'result' key")
            
            candidates = parsed_response["result"]
            if not isinstance(candidates, list):
                logger.error(f"'result' is not a list. Type: {type(candidates)}")
                raise TypeError("'result' must be a list")
            
            # Verify word counts
            for i, candidate in enumerate(candidates):
                candidate_word_count = len(candidate.split())
                if candidate_word_count != word_count:
                    logger.warning(f"Candidate {i+1} has {candidate_word_count} words, expected {word_count}")
            
            if len(candidates) != num_candidates:
                logger.warning(f"Expected {num_candidates} candidates but got {len(candidates)}")
            
            return candidates
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw content that failed to parse: {response.choices[0].message.content}")
            raise
            
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {e}")
        raise

def process_iteration(
    target_text: str,
    target_embedding: List[float],
    previous_results: List[Tuple[str, float, int]] = None,  # Now includes round number
    round_number: int = 1,
    num_candidates: int = 5,
    embedding_model: str = "text-embedding-3-small",
    llm_model: str = "gpt-4",
    abstract_description: str = None,
    temperature: float = 0.5
) -> List[Tuple[str, float]]:
    """Process one iteration of the embedding matching loop."""
    logger.info(f"Starting iteration {round_number}")
    
    # Generate candidate texts
    candidates = generate_candidates(
        target_text, 
        previous_results, 
        round_number,
        num_candidates,
        llm_model,
        abstract_description,
        temperature
    )
    
    # Get embeddings for all candidates at once
    candidate_embeddings = get_embeddings(candidates, embedding_model)
    
    # Calculate differences and create (text, error) pairs
    results = [
        (text, calculate_vector_difference(target_embedding, emb))
        for text, emb in zip(candidates, candidate_embeddings)
    ]
    
    # Sort by error (ascending)
    sorted_results = sorted(results, key=lambda x: x[1])
    logger.info(f"Completed iteration {round_number}. Best error: {sorted_results[0][1]:.4f}")
    
    return sorted_results 