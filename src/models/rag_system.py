"""
RAG (Retrieval-Augmented Generation) system with LangGraph.
Implements a stateful conversation flow with retrieval and generation.
"""

import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.models.llm_config import get_default_llm
from src.models.custom_retriever import BusinessDataRetriever
from src.models.memory_manager import ConversationMemoryManager
from src.utils.prompts import get_rag_prompt


class GraphState(TypedDict):
    """State for the RAG graph."""
    question: str
    context: List[str]
    statistics: str
    chat_history: List[BaseMessage]
    answer: str


class RAGSystem:
    """RAG system with LangGraph for stateful conversations."""

    def __init__(
        self,
        df: pd.DataFrame,
        vector_retriever: Any,
        memory_manager: Optional[ConversationMemoryManager] = None
    ):
        """
        Initialize RAG system.

        Args:
            df: Business data DataFrame
            vector_retriever: Vector-based retriever
            memory_manager: Optional memory manager for conversations
        """
        self.df = df
        self.llm = get_default_llm()
        self.memory_manager = memory_manager or ConversationMemoryManager(k=5)

        # Create custom retriever
        self.retriever = BusinessDataRetriever(
            df=df,
            vector_retriever=vector_retriever,
            k=5
        )

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile
        return workflow.compile()

    def _retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Retrieval node: Get relevant context and statistics.

        Args:
            state: Current graph state

        Returns:
            Updated state with context and statistics
        """
        question = state["question"]

        # Retrieve documents using invoke (modern LangChain method)
        docs = self.retriever.invoke(question)

        # Extract context
        context = [doc.page_content for doc in docs]

        # Generate statistics summary
        statistics = self._generate_statistics_summary(question)

        return {
            "context": context,
            "statistics": statistics
        }

    def _generate_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Generation node: Generate answer using LLM.

        Args:
            state: Current graph state

        Returns:
            Updated state with answer
        """
        # Prepare inputs
        context_text = "\n\n".join(state["context"])
        statistics_text = state["statistics"]
        question = state["question"]
        chat_history = state.get("chat_history", [])

        # Get prompt
        prompt = get_rag_prompt()

        # Create chain
        chain = prompt | self.llm | StrOutputParser()

        # Generate answer
        answer = chain.invoke({
            "context": context_text,
            "statistics": statistics_text,
            "question": question,
            "chat_history": chat_history
        })

        return {"answer": answer}

    def _generate_statistics_summary(self, question: str) -> str:
        """
        Generate a quick statistics summary relevant to the question.

        Args:
            question: User question

        Returns:
            Statistics summary string
        """
        summary_parts = []

        # Always include basic info
        summary_parts.append(f"Total Records: {len(self.df)}")

        # Add relevant statistics based on question
        query_lower = question.lower()

        # Sales statistics
        if 'sales' in query_lower or 'revenue' in query_lower:
            sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
            if sales_cols:
                sales_col = sales_cols[0]
                summary_parts.append(f"Total Sales: ${self.df[sales_col].sum():,.2f}")
                summary_parts.append(f"Average Sale: ${self.df[sales_col].mean():,.2f}")

        # Product statistics
        if 'product' in query_lower:
            product_cols = [col for col in self.df.columns if 'product' in col.lower()]
            if product_cols:
                summary_parts.append(f"Total Products: {self.df[product_cols[0]].nunique()}")

        # Regional statistics
        if 'region' in query_lower or 'location' in query_lower:
            region_cols = [col for col in self.df.columns if 'region' in col.lower()]
            if region_cols:
                summary_parts.append(f"Total Regions: {self.df[region_cols[0]].nunique()}")

        return "\n".join(summary_parts)

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question: User question

        Returns:
            Generated answer
        """
        # Get chat history from memory
        chat_history = self.memory_manager.get_chat_history()

        # Prepare initial state
        initial_state = {
            "question": question,
            "context": [],
            "statistics": "",
            "chat_history": chat_history,
            "answer": ""
        }

        # Run the graph
        result = self.graph.invoke(initial_state)

        # Extract answer
        answer = result["answer"]

        # Update memory
        self.memory_manager.add_exchange(question, answer)

        return answer

    def clear_memory(self):
        """Clear conversation memory."""
        self.memory_manager.clear()

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get conversation history.

        Returns:
            List of messages
        """
        return self.memory_manager.get_chat_history()


class SimpleRAGChain:
    """Simplified RAG chain without LangGraph (alternative implementation)."""

    def __init__(
        self,
        df: pd.DataFrame,
        vector_retriever: Any,
        memory_manager: Optional[ConversationMemoryManager] = None
    ):
        """
        Initialize simple RAG chain.

        Args:
            df: Business data DataFrame
            vector_retriever: Vector-based retriever
            memory_manager: Optional memory manager
        """
        self.df = df
        self.llm = get_default_llm()
        self.memory_manager = memory_manager or ConversationMemoryManager(k=5)

        # Create custom retriever
        self.retriever = BusinessDataRetriever(
            df=df,
            vector_retriever=vector_retriever,
            k=5
        )

        # Get prompt template
        self.prompt = get_rag_prompt()

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question: User question

        Returns:
            Generated answer
        """
        # Retrieve context
        docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate statistics
        statistics = self._generate_statistics(question)

        # Get chat history
        chat_history = self.memory_manager.get_chat_history()

        # Create chain
        chain = self.prompt | self.llm | StrOutputParser()

        # Generate answer
        answer = chain.invoke({
            "context": context,
            "statistics": statistics,
            "question": question,
            "chat_history": chat_history
        })

        # Update memory
        self.memory_manager.add_exchange(question, answer)

        return answer

    def _generate_statistics(self, question: str) -> str:
        """Generate relevant statistics."""
        summary_parts = [f"Total Records: {len(self.df)}"]

        query_lower = question.lower()

        if 'sales' in query_lower or 'revenue' in query_lower:
            sales_cols = [col for col in self.df.columns if 'sales' in col.lower()]
            if sales_cols:
                sales_col = sales_cols[0]
                summary_parts.append(f"Total Sales: ${self.df[sales_col].sum():,.2f}")

        return "\n".join(summary_parts)
