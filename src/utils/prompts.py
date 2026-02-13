"""
Prompt templates and engineering for the BI Assistant.
Contains carefully crafted prompts for different analysis tasks.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# System prompt for the BI Assistant
SYSTEM_PROMPT = """You are an expert Business Intelligence Analyst with deep expertise in data analysis, statistics, and business insights.

Your role is to:
1. Analyze business data and identify meaningful patterns and trends
2. Provide accurate, data-driven insights based on the provided context
3. Answer questions clearly and concisely using natural language
4. Support your answers with specific numbers and statistics from the data
5. Recommend actionable business strategies when appropriate

Guidelines:
- Always base your answers on the provided data context
- Be precise with numbers and statistics
- If you don't have enough information to answer accurately, say so
- Present insights in a clear, business-friendly manner
- When discussing trends, mention specific time periods and values
- For comparisons, provide percentage changes and absolute values"""


# RAG prompt template
RAG_PROMPT_TEMPLATE = """Based on the following business data context, answer the user's question accurately and comprehensively.

Context from the data:
{context}

Current Statistics:
{statistics}

Question: {question}

Instructions:
- Use the provided context and statistics to answer the question
- Cite specific numbers and data points
- If the question asks for trends, describe them with specific values
- If the question asks for comparisons, provide detailed comparisons
- Keep your answer focused and relevant to the question

Answer:"""


# Data analysis prompt
ANALYSIS_PROMPT_TEMPLATE = """Analyze the following business data and provide insights:

Data Summary:
{data_summary}

Analysis Focus: {focus}

Provide a comprehensive analysis that includes:
1. Key findings and patterns
2. Notable trends or anomalies
3. Statistical insights (means, medians, distributions)
4. Business implications
5. Recommendations (if applicable)

Analysis:"""


# Visualization recommendation prompt
VISUALIZATION_PROMPT = """Based on the user's question and the data available, recommend the most appropriate visualization.

Question: {question}

Available data dimensions: {dimensions}

Recommend ONE of the following visualization types:
- line_chart: For trends over time
- bar_chart: For comparing categories or products
- pie_chart: For showing composition or distribution
- scatter_plot: For showing relationships between variables
- heatmap: For regional or multi-dimensional comparisons
- histogram: For showing distribution of values

Respond with just the visualization type and a brief reason (one sentence).

Recommendation:"""


# Query intent classification prompt
INTENT_PROMPT = """Classify the user's question into one of the following categories:

Question: {question}

Categories:
- sales_analysis: Questions about sales performance, revenue, or sales trends
- product_analysis: Questions about specific products or product comparisons
- regional_analysis: Questions about geographic regions or location-based insights
- customer_analysis: Questions about customers, demographics, or segmentation
- time_analysis: Questions about trends over time or specific time periods
- statistical_query: Questions asking for specific statistics (mean, median, etc.)
- general_query: General questions about the business or data

Respond with just the category name.

Category:"""


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get the RAG prompt template.

    Returns:
        ChatPromptTemplate for RAG pipeline
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", RAG_PROMPT_TEMPLATE)
    ])


def get_analysis_prompt() -> ChatPromptTemplate:
    """
    Get the analysis prompt template.

    Returns:
        ChatPromptTemplate for data analysis
    """
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", ANALYSIS_PROMPT_TEMPLATE)
    ])


def get_visualization_prompt() -> ChatPromptTemplate:
    """
    Get the visualization recommendation prompt.

    Returns:
        ChatPromptTemplate for visualization recommendations
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a data visualization expert."),
        ("human", VISUALIZATION_PROMPT)
    ])


def get_intent_prompt() -> ChatPromptTemplate:
    """
    Get the intent classification prompt.

    Returns:
        ChatPromptTemplate for intent classification
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a query classification expert."),
        ("human", INTENT_PROMPT)
    ])


# Conversation prompts
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


# Follow-up question generation
FOLLOWUP_PROMPT = """Based on the analysis provided, suggest 3 relevant follow-up questions the user might want to ask.

Analysis: {analysis}

Generate questions that would provide deeper insights or explore related aspects of the data.

Follow-up Questions:
1.
2.
3. """
