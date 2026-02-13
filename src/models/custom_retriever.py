"""
Custom retriever for extracting relevant statistics from business data.
Combines vector similarity search with structured data queries.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever


class BusinessDataRetriever(BaseRetriever):
    """Custom retriever that extracts relevant statistics from business data."""

    df: pd.DataFrame
    vector_retriever: Any
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents and statistics.

        Args:
            query: User query
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        # Get vector-based results
        # Use invoke() for modern LangChain retrievers (Chroma, etc.)
        try:
            vector_docs = self.vector_retriever.invoke(query)
        except AttributeError:
            # Fallback to get_relevant_documents for older retrievers
            vector_docs = self.vector_retriever.get_relevant_documents(query)

        # Extract intent and generate statistics
        intent = self._classify_intent(query)
        stats_docs = self._generate_statistics_documents(intent, query)

        # Combine results
        all_docs = vector_docs + stats_docs

        return all_docs[:self.k]

    def _classify_intent(self, query: str) -> str:
        """
        Classify the query intent.

        Args:
            query: User query

        Returns:
            Intent category
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ['sales', 'revenue', 'sell', 'sold']):
            return 'sales'
        elif any(word in query_lower for word in ['product', 'item']):
            return 'product'
        elif any(word in query_lower for word in ['region', 'location', 'area', 'geographic']):
            return 'region'
        elif any(word in query_lower for word in ['customer', 'buyer', 'demographics', 'age', 'gender']):
            return 'customer'
        elif any(word in query_lower for word in ['trend', 'time', 'period', 'month', 'year', 'quarter']):
            return 'time'
        elif any(word in query_lower for word in ['average', 'mean', 'median', 'std', 'statistics']):
            return 'statistics'
        else:
            return 'general'

    def _generate_statistics_documents(self, intent: str, query: str) -> List[Document]:
        """
        Generate statistical documents based on intent.

        Args:
            intent: Query intent
            query: Original query

        Returns:
            List of documents with statistics
        """
        documents = []

        if intent == 'sales':
            documents.append(self._get_sales_statistics())
        elif intent == 'product':
            documents.append(self._get_product_statistics())
        elif intent == 'region':
            documents.append(self._get_regional_statistics())
        elif intent == 'customer':
            documents.append(self._get_customer_statistics())
        elif intent == 'time':
            documents.append(self._get_time_statistics())
        elif intent == 'statistics':
            documents.append(self._get_general_statistics())

        return documents

    def _get_sales_statistics(self) -> Document:
        """Get sales-related statistics."""
        sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]

        if not sales_cols:
            return Document(page_content="No sales data available.", metadata={"type": "statistics"})

        sales_col = sales_cols[0]
        stats = self.df[sales_col].describe()

        content = f"""Sales Statistics:
Total Sales: {self.df[sales_col].sum():,.2f}
Average Sale: {stats['mean']:,.2f}
Median Sale: {self.df[sales_col].median():,.2f}
Standard Deviation: {stats['std']:,.2f}
Minimum Sale: {stats['min']:,.2f}
Maximum Sale: {stats['max']:,.2f}
Total Transactions: {len(self.df)}"""

        return Document(page_content=content, metadata={"type": "sales_statistics"})

    def _get_product_statistics(self) -> Document:
        """Get product-related statistics."""
        product_cols = [col for col in self.df.columns if 'product' in col.lower()]

        if not product_cols:
            return Document(page_content="No product data available.", metadata={"type": "statistics"})

        product_col = product_cols[0]
        sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]

        if sales_cols:
            sales_col = sales_cols[0]
            product_sales = self.df.groupby(product_col)[sales_col].agg(['sum', 'mean', 'count'])
            product_sales = product_sales.sort_values('sum', ascending=False)

            content = f"""Product Statistics:
Total Products: {self.df[product_col].nunique()}

Top 5 Products by Sales:
"""
            for idx, (product, row) in enumerate(product_sales.head(5).iterrows(), 1):
                content += f"{idx}. {product}: Total Sales=${row['sum']:,.2f}, Avg=${row['mean']:,.2f}, Count={int(row['count'])}\n"

        else:
            product_counts = self.df[product_col].value_counts()
            content = f"""Product Statistics:
Total Products: {len(product_counts)}

Top 5 Products by Frequency:
"""
            for idx, (product, count) in enumerate(product_counts.head(5).items(), 1):
                content += f"{idx}. {product}: {count} occurrences\n"

        return Document(page_content=content, metadata={"type": "product_statistics"})

    def _get_regional_statistics(self) -> Document:
        """Get regional statistics."""
        region_cols = [col for col in self.df.columns if 'region' in col.lower() or 'location' in col.lower()]

        if not region_cols:
            return Document(page_content="No regional data available.", metadata={"type": "statistics"})

        region_col = region_cols[0]
        sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]

        if sales_cols:
            sales_col = sales_cols[0]
            regional_sales = self.df.groupby(region_col)[sales_col].agg(['sum', 'mean', 'count'])
            regional_sales = regional_sales.sort_values('sum', ascending=False)

            content = f"""Regional Statistics:
Total Regions: {self.df[region_col].nunique()}

Performance by Region:
"""
            for region, row in regional_sales.iterrows():
                content += f"{region}: Total=${row['sum']:,.2f}, Avg=${row['mean']:,.2f}, Count={int(row['count'])}\n"

        else:
            region_counts = self.df[region_col].value_counts()
            content = f"""Regional Statistics:
Total Regions: {len(region_counts)}

Transactions by Region:
"""
            for region, count in region_counts.items():
                content += f"{region}: {count} transactions\n"

        return Document(page_content=content, metadata={"type": "regional_statistics"})

    def _get_customer_statistics(self) -> Document:
        """Get customer demographics and statistics."""
        age_cols = [col for col in self.df.columns if 'age' in col.lower()]
        gender_cols = [col for col in self.df.columns if 'gender' in col.lower()]

        content = "Customer Statistics:\n"

        if age_cols:
            age_col = age_cols[0]
            age_stats = self.df[age_col].describe()
            content += f"""
Age Demographics:
Average Age: {age_stats['mean']:.1f}
Median Age: {self.df[age_col].median():.1f}
Age Range: {age_stats['min']:.0f} - {age_stats['max']:.0f}
"""

        if gender_cols:
            gender_col = gender_cols[0]
            gender_dist = self.df[gender_col].value_counts()
            content += f"\nGender Distribution:\n"
            for gender, count in gender_dist.items():
                pct = (count / len(self.df)) * 100
                content += f"{gender}: {count} ({pct:.1f}%)\n"

        if not age_cols and not gender_cols:
            content += "No customer demographic data available."

        return Document(page_content=content, metadata={"type": "customer_statistics"})

    def _get_time_statistics(self) -> Document:
        """Get time-based statistics."""
        # Look for date columns (they're strings now after normalization)
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]

        if len(date_cols) == 0:
            return Document(page_content="No time-series data available.", metadata={"type": "statistics"})

        date_col = date_cols[0]
        sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]

        # Convert string dates to datetime for analysis
        df_copy = self.df.copy()
        df_copy['_temp_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=['_temp_date'])

        if len(df_copy) == 0:
            return Document(page_content="No valid time-series data available.", metadata={"type": "statistics"})

        content = f"""Time Period Statistics:
Date Range: {df_copy['_temp_date'].min().date()} to {df_copy['_temp_date'].max().date()}
Total Days: {(df_copy['_temp_date'].max() - df_copy['_temp_date'].min()).days}
"""

        if sales_cols:
            sales_col = sales_cols[0]
            # Monthly aggregation
            monthly_sales = df_copy.groupby(df_copy['_temp_date'].dt.to_period('M'))[sales_col].sum()
            content += f"\nMonthly Sales Summary:\n"
            content += f"Average Monthly Sales: ${monthly_sales.mean():,.2f}\n"
            content += f"Best Month: {monthly_sales.idxmax()} (${monthly_sales.max():,.2f})\n"
            content += f"Worst Month: {monthly_sales.idxmin()} (${monthly_sales.min():,.2f})\n"

        return Document(page_content=content, metadata={"type": "time_statistics"})

    def _get_general_statistics(self) -> Document:
        """Get general statistical measures."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        content = "General Statistics:\n\n"

        for col in numeric_cols:
            stats = self.df[col].describe()
            content += f"""{col}:
Mean: {stats['mean']:.2f}
Median: {self.df[col].median():.2f}
Std Dev: {stats['std']:.2f}
Min: {stats['min']:.2f}
Max: {stats['max']:.2f}

"""

        return Document(page_content=content, metadata={"type": "general_statistics"})
