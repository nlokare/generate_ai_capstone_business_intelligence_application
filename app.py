"""
InsightForge: AI-Powered Business Intelligence Assistant
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv

from src.data.loader import DataLoader
from src.data.knowledge_base import KnowledgeBase
from src.models.rag_system import RAGSystem
from src.models.memory_manager import ConversationMemoryManager
from src.visualization.charts import BusinessVisualizer
from src.utils.evaluation import ModelEvaluator, generate_test_questions

# Load environment variables - find .env file automatically
env_file = find_dotenv()
if env_file:
    load_dotenv(env_file, override=True)  # Use override=True to ensure it loads
else:
    # Try loading from current directory
    load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="InsightForge - AI Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False  # Set to True only when RAG system is ready
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'memory_manager' not in st.session_state:
        st.session_state.memory_manager = ConversationMemoryManager(k=5)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = None
    if 'initialization_error' not in st.session_state:
        st.session_state.initialization_error = None  # Store error messages


def load_data(uploaded_file):
    """Load data from uploaded file."""
    try:
        with st.spinner("Loading and processing data..."):
            # Save uploaded file temporarily
            temp_path = Path("data/raw/temp_upload.csv")
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load data
            loader = DataLoader()
            df = loader.load_data(str(temp_path))

            # Validate data
            issues = loader.validate_data()
            if issues:
                st.warning(f"Data validation warnings: {', '.join(issues)}")

            # Store data but DON'T set data_loaded yet - wait for full initialization
            st.session_state.df = df

            # Create visualizer
            st.session_state.visualizer = BusinessVisualizer(df)

            return df, loader.get_metadata()

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


def create_knowledge_base(df):
    """Create knowledge base from data using Chroma."""
    try:
        with st.spinner("Creating knowledge base... This may take a minute."):
            # Create knowledge base with Chroma
            kb = KnowledgeBase()
            vector_store = kb.create_from_dataframe(df)

            st.session_state.knowledge_base = kb

            return kb

    except Exception as e:
        error_msg = str(e)
        st.error(f"Error creating knowledge base: {error_msg}")

        # Check if it's an API key error
        if "api_key" in error_msg.lower() or "openai" in error_msg.lower():
            st.warning("This appears to be an API key issue")
            import os
            if not os.getenv('OPENAI_API_KEY'):
                st.error("OPENAI_API_KEY is not set in the environment")
                st.info("Please ensure your .env file contains: OPENAI_API_KEY=your-key-here")
            else:
                st.info("API key appears to be set. There may be an issue with the key itself.")

        import traceback
        with st.expander("See full error details"):
            st.code(traceback.format_exc())
        return None


def initialize_rag_system(df, knowledge_base):
    """Initialize RAG system."""
    try:
        with st.spinner("Initializing AI system..."):
            retriever = knowledge_base.get_retriever(k=5)

            rag_system = RAGSystem(
                df=df,
                vector_retriever=retriever,
                memory_manager=st.session_state.memory_manager
            )

            st.session_state.rag_system = rag_system

            return rag_system

    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">📊 InsightForge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Business Intelligence Assistant</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")

        uploaded_file = st.file_uploader(
            "Upload your business data (CSV)",
            type=['csv'],
            help="Upload a CSV file with your business data"
        )

        if uploaded_file is not None and not st.session_state.data_loaded:
            df, metadata = load_data(uploaded_file)

            if df is not None:
                st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")

                # Create knowledge base
                kb = create_knowledge_base(df)

                if kb is not None:
                    st.success("✅ Knowledge base created")

                    # Initialize RAG system
                    rag = initialize_rag_system(df, kb)

                    if rag is not None:
                        st.success("✅ AI system ready")

                        # Only set data_loaded after EVERYTHING succeeds
                        st.session_state.data_loaded = True
                        st.session_state.initialization_error = None
                    else:
                        st.error("❌ Failed to initialize AI system. Chat features will not be available.")
                        st.session_state.initialization_error = "RAG system initialization failed"
                        st.info("📊 You can still use the Data Explorer and Visualizations tabs")
                else:
                    st.error("❌ Failed to create knowledge base.")
                    st.session_state.initialization_error = "Knowledge base creation failed - check OPENAI_API_KEY"
                    st.info("📊 You can still use the Data Explorer and Visualizations tabs")

        # Show data info if we have a DataFrame (even if full initialization failed)
        if st.session_state.df is not None:
            st.divider()
            st.header("📊 Data Info")
            st.metric("Total Rows", len(st.session_state.df))
            st.metric("Total Columns", len(st.session_state.df.columns))

            # Show system status
            if st.session_state.rag_system:
                st.success("🤖 AI System: Ready")
            else:
                st.error("🤖 AI System: Not Available")

            st.divider()

            if st.button("🗑️ Clear Data", width="stretch"):
                for key in ['data_loaded', 'df', 'knowledge_base', 'rag_system', 'chat_history', 'visualizer']:
                    if key in st.session_state:
                        st.session_state[key] = None if key != 'data_loaded' else False
                        if key == 'chat_history':
                            st.session_state[key] = []
                st.rerun()

            if st.button("🔄 Clear Chat", width="stretch"):
                st.session_state.chat_history = []
                if st.session_state.memory_manager:
                    st.session_state.memory_manager.clear()
                st.rerun()

    # Main content
    # Show content if data is loaded OR if we have a DataFrame (even if RAG system failed)
    has_data = st.session_state.df is not None

    if not has_data:
        st.info("👈 Please upload a CSV file to get started")

        # Display example
        with st.expander("📋 Example Data Format"):
            st.markdown("""
            Your CSV file should include columns such as:
            - **Date**: Transaction date (e.g., 2024-01-01)
            - **Product**: Product name
            - **Sales/Revenue**: Sales amount
            - **Region**: Geographic region
            - **Customer demographics**: Age, Gender, etc.

            Example:
            ```
            Date,Product,Region,Sales,CustomerAge,CustomerGender
            2024-01-01,Product A,North,1500,35,M
            2024-01-02,Product B,South,2300,42,F
            ```
            """)

        return

    # Show warning if initialization failed but data is available
    if st.session_state.initialization_error:
        st.warning(f"⚠️ {st.session_state.initialization_error}")
        st.info("💡 The Chat Assistant and Model Evaluation features require a fully initialized system. Please ensure your OPENAI_API_KEY is set correctly.")

    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Assistant", "📈 Visualizations", "📊 Data Explorer", "🔍 Model Evaluation"])

    # Tab 1: Chat Assistant
    with tab1:
        st.header("Ask Questions About Your Data")

        # Check if RAG system is ready
        if not st.session_state.rag_system:
            st.warning("⚠️ Chat Assistant is not available")

            if st.session_state.initialization_error:
                st.error(f"Error: {st.session_state.initialization_error}")

                if "OPENAI_API_KEY" in st.session_state.initialization_error:
                    st.markdown("""
                    ### How to fix:

                    1. Set your OpenAI API key:
                       ```bash
                       export OPENAI_API_KEY='your-api-key-here'
                       ```

                    2. Restart the Streamlit app:
                       ```bash
                       streamlit run app.py
                       ```

                    3. Upload your data again
                    """)
            else:
                st.info("Please upload data to use the chat assistant.")

        else:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Chat input
        user_question = st.chat_input("Ask a question about your business data...", disabled=not st.session_state.rag_system)

        if user_question:
            if not st.session_state.rag_system:
                st.error("Chat assistant is not available. See the message above for details.")
            else:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_question})

                with st.chat_message("user"):
                    st.write(user_question)

                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_system.query(user_question)
                        st.write(response)

                        # Add to history
                        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Suggested questions
        if len(st.session_state.chat_history) == 0:
            st.divider()
            st.subheader("💡 Suggested Questions")

            col1, col2 = st.columns(2)

            suggestions = [
                "What is the total sales?",
                "Which are the top-selling products?",
                "Show me the sales trend",
                "Which region performs best?",
                "What is the customer demographic?",
                "What is the average order value?"
            ]

            for i, suggestion in enumerate(suggestions):
                col = col1 if i % 2 == 0 else col2
                if col.button(suggestion, key=f"suggest_{i}", width="stretch"):
                    if st.session_state.rag_system:
                        st.session_state.chat_history.append({"role": "user", "content": suggestion})
                        with st.spinner("Thinking..."):
                            response = st.session_state.rag_system.query(suggestion)
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.rerun()
                    else:
                        st.error("Please upload data first to use the assistant.")

    # Tab 2: Visualizations
    with tab2:
        st.header("Data Visualizations")

        viz_option = st.selectbox(
            "Select Visualization Type",
            ["Sales Trend", "Product Comparison", "Regional Analysis", "Customer Demographics", "Summary Dashboard"]
        )

        try:
            if viz_option == "Sales Trend":
                fig = st.session_state.visualizer.create_sales_trend()
                st.plotly_chart(fig, width="stretch")

            elif viz_option == "Product Comparison":
                top_n = st.slider("Number of products to show", 5, 20, 10)
                fig = st.session_state.visualizer.create_product_comparison(top_n=top_n)
                st.plotly_chart(fig, width="stretch")

            elif viz_option == "Regional Analysis":
                fig = st.session_state.visualizer.create_regional_analysis()
                st.plotly_chart(fig, width="stretch")

            elif viz_option == "Customer Demographics":
                fig = st.session_state.visualizer.create_customer_demographics()
                st.plotly_chart(fig, width="stretch")

            elif viz_option == "Summary Dashboard":
                fig = st.session_state.visualizer.create_summary_dashboard()
                st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("This visualization may not be available for your dataset. Try another one.")

    # Tab 3: Data Explorer
    with tab3:
        st.header("Explore Your Data")

        # Display dataframe
        st.subheader("Raw Data")
        # Data is already normalized by the DataLoader - no conversion needed
        # All columns are either numeric, string, or have been converted to consistent types
        st.dataframe(st.session_state.df, width="stretch")

        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(st.session_state.df.describe(), width="stretch")

        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Type': [str(dtype) for dtype in st.session_state.df.dtypes.values],
            'Non-Null Count': st.session_state.df.count().values,
            'Null Count': st.session_state.df.isnull().sum().values
        })
        st.dataframe(col_info, width="stretch")

    # Tab 4: Model Evaluation
    with tab4:
        st.header("Model Performance Evaluation")

        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                try:
                    # Generate test questions
                    test_questions = generate_test_questions(st.session_state.df)

                    # Get responses
                    results = []
                    progress_bar = st.progress(0)

                    for i, question in enumerate(test_questions[:10]):  # Limit to 10 for demo
                        response = st.session_state.rag_system.query(question)
                        results.append({
                            "Question": question,
                            "Response": response,
                            "Response Length": len(response)
                        })
                        progress_bar.progress((i + 1) / min(10, len(test_questions)))

                    # Display results
                    st.success("Evaluation complete!")

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, width="stretch")

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Questions Tested", len(results))
                    col2.metric("Avg Response Length", f"{results_df['Response Length'].mean():.0f} chars")
                    col3.metric("Total Responses", len(results))

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

        st.info("Click 'Run Evaluation' to test the model with sample questions")


if __name__ == "__main__":
    # Force reload of .env file to ensure API key is available
    # This is needed because Streamlit may not pick up environment variables correctly
    from dotenv import load_dotenv, find_dotenv
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file, override=True)

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OpenAI API key not found")
        st.info("Please create a `.env` file with: OPENAI_API_KEY=your-key-here")

        # Show if .env exists but key is missing
        if env_file:
            st.warning(f".env file found at {env_file} but OPENAI_API_KEY not loaded")
            st.info("Make sure the .env file contains: OPENAI_API_KEY=sk-your-key (no quotes, no spaces)")
        else:
            st.warning("No .env file found")

        st.stop()

    main()
