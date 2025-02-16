import streamlit as st
import time
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from embedding_utils import get_embeddings, process_iteration, get_abstract_description

# Available models
EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
]

LLM_MODELS = [
    "gpt-4o-mini",
    "gpt-4o"
]

# Set page config
st.set_page_config(
    page_title="Embedding Decoder",
    page_icon="🔄",
    layout="wide"
)

# Title and description
st.title("🔄 Embedding Decoder")
st.markdown("""
This application attempts to find text snippets that match the embedding of your input text.
Enter a text snippet below and watch as the system iteratively generates and refines candidates
to match the semantic meaning of your text.
""")

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'all_candidates' not in st.session_state:
    st.session_state.all_candidates = []  # Store all (text, error, round) tuples
if 'target_embedding' not in st.session_state:
    st.session_state.target_embedding = None
if 'target_text' not in st.session_state:
    st.session_state.target_text = None
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
if 'error_history' not in st.session_state:
    st.session_state.error_history = {'best': [], 'average': []}
if 'log_data' not in st.session_state:
    st.session_state.log_data = []
if 'abstract_description' not in st.session_state:
    st.session_state.abstract_description = None

# Sidebar for input and settings
with st.sidebar:
    st.header("Input & Settings")
    
    with st.form("input_form"):
        input_text = st.text_area(
            "Target Text:",
            height=100,
            placeholder="Enter a text snippet to find matching embeddings..."
        )
        
        st.subheader("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model:",
            options=EMBEDDING_MODELS,
            index=0
        )
        llm_model = st.selectbox(
            "LLM Model:",
            options=LLM_MODELS,
            index=0
        )
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
            help="Controls randomness in generation. Higher values (>1) make output more random, lower values (<1) make it more focused."
        )
        
        st.subheader("Process Settings")
        max_rounds = st.number_input(
            "Maximum rounds:",
            min_value=1,
            max_value=20,
            value=10
        )
        error_threshold = st.number_input(
            "Error threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Stop when error is below this value"
        )
        num_candidates = st.number_input(
            "Candidates per round:",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of candidates to generate in each round"
        )
        
        submitted = st.form_submit_button("Start Process")

# Main content area
if submitted and input_text:
    # Reset state
    st.session_state.running = True
    st.session_state.results_history = []
    st.session_state.all_candidates = []  # Reset all candidates
    st.session_state.target_text = input_text
    st.session_state.current_round = 0
    st.session_state.error_history = {'best': [], 'average': []}
    st.session_state.log_data = []
    
    with st.spinner("Initializing..."):
        # Get abstract description first
        st.session_state.abstract_description = get_abstract_description(input_text, llm_model)
        # Then get embedding
        st.session_state.target_embedding = get_embeddings([input_text], embedding_model)[0]
    st.rerun()

# Create two columns for the dashboard with more space for the chart
col1, col2 = st.columns([2, 3])

with col1:
    # Log Section
    st.subheader("Log")
    if st.session_state.log_data:
        # Convert log data to DataFrame
        df = pd.DataFrame(st.session_state.log_data, columns=['Round', 'Error', 'Best Candidate'])
        # Style the DataFrame
        st.dataframe(
            df,
            column_config={
                "Round": st.column_config.NumberColumn(
                    "Round",
                    help="Iteration round",
                    format="%d"
                ),
                "Error": st.column_config.NumberColumn(
                    "Error",
                    help="Best error score",
                    format="%.4f"
                ),
                "Best Candidate": st.column_config.TextColumn(
                    "Best Candidate",
                    help="Candidate with lowest error",
                    max_chars=50
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Show final comparison if process is complete
        if not st.session_state.running and st.session_state.log_data:
            st.markdown("---")
            
            # Show completion message
            if st.session_state.error_history['best'][-1] < error_threshold:
                st.success(f"✨ Found a match below error threshold! (Error: {st.session_state.error_history['best'][-1]:.4f})")
            else:
                st.info("✋ Maximum rounds reached!")
            
            st.markdown("### Final Results")
            
            # Get the best candidate (lowest error) from all attempts
            best_candidate = min(st.session_state.all_candidates, key=lambda x: x[1])
            
            col1_1, col1_2 = st.columns(2)
            
            with col1_1:
                st.markdown("**Target Text:**")
                st.info(st.session_state.target_text)
            
            with col1_2:
                st.markdown(f"**Best Candidate (Error: {best_candidate[1]:.4f}):**")
                st.info(best_candidate[0])

with col2:
    # Error Chart
    st.subheader("Error Chart")
    if st.session_state.error_history['best']:
        fig = go.Figure()
        x = list(range(1, len(st.session_state.error_history['best']) + 1))
        
        # Add best error line
        fig.add_trace(go.Scatter(
            x=x,
            y=st.session_state.error_history['best'],
            mode='lines+markers',
            name='Best Error',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Add average error line
        fig.add_trace(go.Scatter(
            x=x,
            y=st.session_state.error_history['average'],
            mode='lines+markers',
            name='Average Error',
            line=dict(color='#3498db', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            xaxis_title='Round',
            yaxis_title='Error Score',
            hovermode='x unified',
            showlegend=True,
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Show abstract description if available
    if st.session_state.abstract_description:
        st.markdown("**Target Description:**")
        st.info(st.session_state.abstract_description)

# Process and display results
if st.session_state.running:
    if st.session_state.current_round < max_rounds:
        st.session_state.current_round += 1
        
        # Process current iteration
        results = process_iteration(
            st.session_state.target_text,
            st.session_state.target_embedding,
            st.session_state.all_candidates,  # Pass all historical results
            st.session_state.current_round,
            num_candidates,
            embedding_model,
            llm_model,
            st.session_state.abstract_description,
            temperature
        )
        
        # Add new results to all_candidates with round number
        for text, error in results:
            st.session_state.all_candidates.append((text, error, st.session_state.current_round))
        
        st.session_state.results_history.append(results)
        
        # Update error history
        best_error = results[0][1]
        avg_error = np.mean([error for _, error in results])
        st.session_state.error_history['best'].append(best_error)
        st.session_state.error_history['average'].append(avg_error)
        
        # Add log data
        st.session_state.log_data.append([
            st.session_state.current_round,
            best_error,
            results[0][0]
        ])
        
        # Check if we've reached the error threshold
        if best_error < error_threshold:
            st.session_state.running = False
            st.rerun()
        
        time.sleep(0.5)  # Small delay for better UX
        st.rerun()
    else:
        st.session_state.running = False
        st.rerun()

# Stop button
if st.session_state.running:
    if st.button("Stop Process", type="secondary"):
        st.session_state.running = False
        st.rerun() 