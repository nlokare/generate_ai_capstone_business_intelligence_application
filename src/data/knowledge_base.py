"""
Knowledge base creation and management module.
Uses Chroma vector database for efficient retrieval.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path
import shutil

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class KnowledgeBase:
    """Creates and manages a vector-based knowledge base from business data using Chroma."""

    def __init__(self, embedding_model: str = "text-embedding-3-small", persist_directory: str = "./chroma_db"):
        """
        Initialize the knowledge base.

        Args:
            embedding_model: OpenAI embedding model to use
            persist_directory: Directory to persist Chroma database
        """
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self.documents: List[Document] = []
        self.df: Optional[pd.DataFrame] = None
        self.persist_directory = persist_directory

    def create_from_dataframe(
        self,
        df: pd.DataFrame,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Create knowledge base from a pandas DataFrame.

        Args:
            df: Input DataFrame
            chunk_size: Size of text chunks for vectorization
            chunk_overlap: Overlap between chunks

        Returns:
            Chroma vector store
        """
        self.df = df

        # Convert DataFrame to documents
        self.documents = self._dataframe_to_documents(df)

        # Create text splitter for large documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # Split documents if needed
        split_docs = text_splitter.split_documents(self.documents)

        # Clean up old database if it exists
        if Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory)

        # Create Chroma vector store
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        return self.vector_store

    def _dataframe_to_documents(self, df: pd.DataFrame) -> List[Document]:
        """
        Convert DataFrame rows to LangChain Documents.

        Args:
            df: Input DataFrame

        Returns:
            List of Documents
        """
        documents = []

        # Create summary document
        summary_text = self._create_summary_text(df)
        documents.append(Document(
            page_content=summary_text,
            metadata={"type": "summary", "source": "dataframe"}
        ))

        # Create documents for data segments
        # Look for date columns (now they're strings after normalization)
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols:
            documents.extend(self._create_time_based_documents(df, date_cols[0]))

        # Create product-based documents if product column exists
        product_cols = [col for col in df.columns if 'product' in col.lower()]
        if product_cols:
            documents.extend(self._create_product_documents(df, product_cols[0]))

        # Create region-based documents if region column exists
        region_cols = [col for col in df.columns if 'region' in col.lower()]
        if region_cols:
            documents.extend(self._create_region_documents(df, region_cols[0]))

        return documents

    def _create_summary_text(self, df: pd.DataFrame) -> str:
        """Create a summary text of the dataset."""
        summary_parts = [
            f"Dataset Overview:",
            f"Total Records: {len(df)}",
            f"Columns: {', '.join(df.columns)}",
            f"\nNumeric Columns Statistics:"
        ]

        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats = df[col].describe()
            summary_parts.append(
                f"{col}: Mean={stats['mean']:.2f}, Median={df[col].median():.2f}, "
                f"Std={stats['std']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}"
            )

        # Add categorical column info (now includes date strings)
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            summary_parts.append("\nText/Categorical Columns:")
            for col in text_cols:
                unique_values = df[col].nunique()
                top_values = df[col].value_counts().head(3)
                summary_parts.append(
                    f"{col}: {unique_values} unique values. "
                    f"Top values: {', '.join([f'{v} ({c})' for v, c in top_values.items()])}"
                )

        return "\n".join(summary_parts)

    def _create_time_based_documents(
        self,
        df: pd.DataFrame,
        date_col: str
    ) -> List[Document]:
        """
        Create documents grouped by time periods.

        Note: Dates are now string format (YYYY-MM-DD) after normalization.
        """
        documents = []

        try:
            # Convert string dates back to datetime for grouping
            df_copy = df.copy()
            df_copy['_temp_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')

            # Group by month
            df_copy['YearMonth'] = df_copy['_temp_date'].dt.to_period('M')

            for period, group in df_copy.groupby('YearMonth'):
                if pd.notna(period):  # Skip NaT periods
                    text = self._create_group_summary(
                        group,
                        f"Time Period: {period}",
                        exclude_cols=[date_col, 'YearMonth', '_temp_date']
                    )
                    documents.append(Document(
                        page_content=text,
                        metadata={"type": "time_period", "period": str(period)}
                    ))
        except Exception as e:
            # If time-based grouping fails, skip it
            print(f"Warning: Could not create time-based documents: {e}")

        return documents

    def _create_product_documents(
        self,
        df: pd.DataFrame,
        product_col: str
    ) -> List[Document]:
        """Create documents grouped by product."""
        documents = []

        for product, group in df.groupby(product_col):
            if product and str(product).strip():  # Skip empty products
                text = self._create_group_summary(
                    group,
                    f"Product: {product}",
                    exclude_cols=[product_col]
                )
                documents.append(Document(
                    page_content=text,
                    metadata={"type": "product", "product": str(product)}
                ))

        return documents

    def _create_region_documents(
        self,
        df: pd.DataFrame,
        region_col: str
    ) -> List[Document]:
        """Create documents grouped by region."""
        documents = []

        for region, group in df.groupby(region_col):
            if region and str(region).strip():  # Skip empty regions
                text = self._create_group_summary(
                    group,
                    f"Region: {region}",
                    exclude_cols=[region_col]
                )
                documents.append(Document(
                    page_content=text,
                    metadata={"type": "region", "region": str(region)}
                ))

        return documents

    def _create_group_summary(
        self,
        group: pd.DataFrame,
        title: str,
        exclude_cols: List[str] = []
    ) -> str:
        """Create a summary text for a data group."""
        summary_parts = [title, f"Records: {len(group)}"]

        # Numeric summaries
        numeric_cols = [
            col for col in group.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

        for col in numeric_cols:
            total = group[col].sum()
            avg = group[col].mean()
            summary_parts.append(f"{col}: Total={total:.2f}, Average={avg:.2f}")

        return "\n".join(summary_parts)

    def save(self, directory: str = None):
        """
        Save the knowledge base to disk.

        Args:
            directory: Directory to save the knowledge base (uses persist_directory if None)
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save")

        # Chroma automatically persists to persist_directory
        # No additional save needed
        print(f"Vector store persisted to {self.persist_directory}")

    def load(self, directory: str = None):
        """
        Load the knowledge base from disk.

        Args:
            directory: Directory containing the saved knowledge base (uses persist_directory if None)
        """
        load_dir = directory or self.persist_directory
        path = Path(load_dir)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {load_dir}")

        # Load Chroma vector store
        self.vector_store = Chroma(
            persist_directory=load_dir,
            embedding_function=self.embeddings
        )

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")

        return self.vector_store.similarity_search(query, k=k)

    def get_retriever(self, k: int = 5):
        """
        Get a retriever for the knowledge base.

        Args:
            k: Number of documents to retrieve

        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("No vector store available")

        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def cleanup(self):
        """Clean up the persisted Chroma database."""
        if Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory)
            print(f"Cleaned up database at {self.persist_directory}")
