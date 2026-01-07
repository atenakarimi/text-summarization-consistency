"""
Text Summarization Consistency Analyzer - Streamlit Application

A scientific tool for measuring and analyzing the consistency of text
summarization algorithms across multiple runs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.consistency import run_consistency_experiment
from algorithms.extractive import get_available_algorithms
from utils.data import (
    load_sample_articles,
    get_all_titles,
    get_article_by_title,
    get_text_statistics
)
from utils.metrics import (
    calculate_consistency_metrics,
    calculate_length_stats
)


# Page configuration
st.set_page_config(
    page_title="Text Summarization Consistency Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1558a0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = None
    if "selected_article" not in st.session_state:
        st.session_state.selected_article = None


def create_similarity_heatmap(summaries):
    """Create heatmap showing pairwise similarity between summaries."""
    import numpy as np
    from utils.metrics import calculate_jaccard_similarity
    
    n = len(summaries)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                sim = calculate_jaccard_similarity(summaries[i], summaries[j])
                similarity_matrix[i][j] = sim
    
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Run", y="Run", color="Similarity"),
        x=[f"Run {i+1}" for i in range(n)],
        y=[f"Run {i+1}" for i in range(n)],
        color_continuous_scale="Blues",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Pairwise Similarity Between Summaries",
        height=400
    )
    
    return fig


def create_length_distribution_chart(lengths):
    """Create histogram of summary lengths."""
    fig = px.histogram(
        x=lengths,
        nbins=20,
        labels={"x": "Summary Length (characters)", "y": "Frequency"},
        title="Distribution of Summary Lengths"
    )
    
    fig.update_traces(marker_color="#1f77b4")
    fig.update_layout(height=350)
    
    return fig


def create_consistency_gauge(consistency_score):
    """Create gauge chart for consistency score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=consistency_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Consistency Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 33], 'color': "#ffcccc"},
                {'range': [33, 66], 'color': "#ffffcc"},
                {'range': [66, 100], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=250)
    
    return fig


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 Text Summarization Consistency Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Measure reproducibility and consistency of summarization algorithms</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Experiment Configuration")
        
        # Load articles
        df_articles = load_sample_articles()
        
        if df_articles.empty:
            st.error("No sample articles found. Please check data/sample_articles.csv")
            st.stop()
        
        # Article selection
        article_titles = get_all_titles()
        selected_title = st.selectbox(
            "Select Article",
            options=article_titles,
            help="Choose an article to analyze"
        )
        
        # Algorithm selection
        available_algorithms = get_available_algorithms()
        algorithm_display = {
            "textrank": "TextRank (Graph-based)",
            "lexrank": "LexRank (Eigenvector)",
            "luhn": "Luhn (Keyword-based)"
        }
        
        selected_algorithm = st.radio(
            "Select Algorithm",
            options=available_algorithms,
            format_func=lambda x: algorithm_display.get(x, x),
            help="Choose the summarization algorithm to test"
        )
        
        # Experiment parameters
        st.markdown("---")
        st.markdown("### 🔬 Experiment Parameters")
        
        num_runs = st.slider(
            "Number of Runs",
            min_value=5,
            max_value=20,
            value=10,
            step=1,
            help="How many times to run the algorithm"
        )
        
        num_sentences = st.slider(
            "Summary Length (sentences)",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Number of sentences to extract"
        )
        
        # Run button
        st.markdown("---")
        run_button = st.button("🚀 Run Experiment", use_container_width=True)
        
        # About section
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This tool measures how **consistent** summarization algorithms are 
        when run multiple times on the same text.
        
        **Key Metrics:**
        - Consistency Score (0-100%)
        - Unique Output Count
        - Average Similarity
        - Length Variance
        """)
    
    # Main content area
    if run_button:
        article_data = get_article_by_title(selected_title)
        
        if article_data is None:
            st.error("Article not found!")
            st.stop()
        
        article_content = article_data["content"]
        
        with st.spinner(f"Running {num_runs} experiments with {selected_algorithm.upper()}..."):
            # Run experiment
            results = run_consistency_experiment(
                text=article_content,
                algorithm=selected_algorithm,
                num_runs=num_runs,
                num_sentences=num_sentences
            )
            
            st.session_state.experiment_results = results
            st.session_state.selected_article = article_data
    
    # Display results if available
    if st.session_state.experiment_results is not None:
        results = st.session_state.experiment_results
        article_data = st.session_state.selected_article
        
        # Article information
        st.markdown("### 📝 Selected Article")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Title:** {article_data['title']}")
            st.markdown(f"**Category:** {article_data['category']}")
        
        with col2:
            text_stats = get_text_statistics(article_data["content"])
            st.metric("Words", text_stats["word_count"])
        
        # Original text (collapsible)
        with st.expander("📄 View Original Text"):
            st.text(article_data["content"])
        
        st.markdown("---")
        
        # Consistency Metrics
        st.markdown("### 📊 Consistency Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Consistency Score",
                f"{results['consistency_score']:.1f}%",
                help="100% = perfectly consistent, 0% = completely random"
            )
        
        with col2:
            st.metric(
                "Unique Outputs",
                f"{results['unique_summaries']}/{results['summaries'].__len__()}",
                help="Number of different summaries generated"
            )
        
        with col3:
            st.metric(
                "Avg Length",
                f"{results['avg_length']:.0f} chars",
                help="Average summary length"
            )
        
        with col4:
            is_deterministic = results['unique_summaries'] == 1
            st.metric(
                "Deterministic",
                "Yes" if is_deterministic else "No",
                help="Does it always produce the same output?"
            )
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Visualizations")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Consistency Gauge", "📊 Similarity Heatmap", "📏 Length Distribution"])
        
        with tab1:
            fig_gauge = create_consistency_gauge(results['consistency_score'])
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if results['consistency_score'] == 100:
                st.success("✅ Perfect consistency! Algorithm is fully deterministic.")
            elif results['consistency_score'] >= 80:
                st.info("ℹ️ High consistency. Minor variations in output.")
            elif results['consistency_score'] >= 50:
                st.warning("⚠️ Moderate consistency. Noticeable variations.")
            else:
                st.error("❌ Low consistency. Highly variable outputs.")
        
        with tab2:
            if len(results['summaries']) > 1:
                fig_heatmap = create_similarity_heatmap(results['summaries'])
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("Need at least 2 runs to show similarity heatmap")
        
        with tab3:
            fig_length = create_length_distribution_chart(results['lengths'])
            st.plotly_chart(fig_length, use_container_width=True)
            
            length_stats = calculate_length_stats(results['summaries'])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Length", f"{length_stats['min_length']} chars")
            with col2:
                st.metric("Max Length", f"{length_stats['max_length']} chars")
            with col3:
                st.metric("Std Dev", f"{length_stats['std_length']:.1f}")
        
        st.markdown("---")
        
        # Generated Summaries
        st.markdown("### 📋 Generated Summaries")
        
        # Show first 5 summaries by default, with option to expand
        num_to_show = min(5, len(results['summaries']))
        
        for i in range(num_to_show):
            with st.expander(f"Run {i+1} - Length: {results['lengths'][i]} chars"):
                st.markdown(f'<div class="summary-box">{results["summaries"][i]}</div>', unsafe_allow_html=True)
        
        if len(results['summaries']) > num_to_show:
            if st.button(f"Show all {len(results['summaries'])} summaries"):
                for i in range(num_to_show, len(results['summaries'])):
                    with st.expander(f"Run {i+1} - Length: {results['lengths'][i]} chars"):
                        st.markdown(f'<div class="summary-box">{results["summaries"][i]}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Export results
        st.markdown("### 💾 Export Results")
        
        # Prepare data for export
        export_df = pd.DataFrame({
            "Run": [i+1 for i in range(len(results['summaries']))],
            "Summary": results['summaries'],
            "Length": results['lengths']
        })
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"consistency_results_{selected_algorithm}.csv",
            mime="text/csv"
        )
    
    else:
        # Welcome message
        st.info("👈 Configure your experiment in the sidebar and click **Run Experiment** to begin!")
        
        st.markdown("### 🎯 What does this tool do?")
        st.markdown("""
        This application helps you understand the **reproducibility** of different text summarization algorithms by:
        
        1. **Running the same algorithm multiple times** on identical input
        2. **Measuring consistency** in the outputs
        3. **Visualizing variations** through interactive charts
        4. **Comparing determinism** across different algorithms
        
        This is useful for:
        - 🔬 Research on algorithm stability
        - 📊 Understanding NLP reproducibility
        - 🤖 Evaluating summarization methods
        - 📚 Educational purposes
        """)
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Select an article from the sidebar
        2. Choose a summarization algorithm
        3. Set experiment parameters (runs & summary length)
        4. Click **Run Experiment**
        5. Analyze the results!
        """)


if __name__ == "__main__":
    main()
