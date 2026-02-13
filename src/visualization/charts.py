"""
Data visualization module.
Creates various charts and visualizations for business insights.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class BusinessVisualizer:
    """Creates visualizations for business data analysis."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize visualizer.

        Args:
            df: Business data DataFrame
        """
        self.df = df
        self.color_scheme = px.colors.qualitative.Set2

    def create_sales_trend(
        self,
        date_col: str = None,
        sales_col: str = None,
        period: str = 'M'
    ) -> go.Figure:
        """
        Create sales trend line chart.

        Args:
            date_col: Date column name (auto-detected if None)
            sales_col: Sales column name (auto-detected if None)
            period: Aggregation period ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            Plotly figure
        """
        # Auto-detect columns
        if date_col is None:
            # Look for columns with 'date' in the name (they're now strings after normalization)
            date_cols = [col for col in self.df.columns if 'date' in col.lower()]
            if len(date_cols) == 0:
                raise ValueError("No date column found")
            date_col = date_cols[0]

        if sales_col is None:
            sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
            if not sales_cols:
                raise ValueError("No sales column found")
            sales_col = sales_cols[0]

        # Aggregate by period
        df_copy = self.df.copy()
        # Convert string dates back to datetime for grouping
        df_copy['_temp_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy['Period'] = df_copy['_temp_date'].dt.to_period(period)
        sales_by_period = df_copy.groupby('Period')[sales_col].sum().reset_index()
        sales_by_period['Period'] = sales_by_period['Period'].astype(str)

        # Create figure
        fig = px.line(
            sales_by_period,
            x='Period',
            y=sales_col,
            title=f'Sales Trend Over Time ({period})',
            labels={'Period': 'Time Period', sales_col: 'Sales ($)'},
            markers=True
        )

        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )

        return fig

    def create_product_comparison(
        self,
        product_col: str = None,
        value_col: str = None,
        top_n: int = 10
    ) -> go.Figure:
        """
        Create product performance comparison bar chart.

        Args:
            product_col: Product column name (auto-detected if None)
            value_col: Value column name (auto-detected if None)
            top_n: Number of top products to show

        Returns:
            Plotly figure
        """
        # Auto-detect columns
        if product_col is None:
            product_cols = [col for col in self.df.columns if 'product' in col.lower()]
            if not product_cols:
                raise ValueError("No product column found")
            product_col = product_cols[0]

        if value_col is None:
            value_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
            if not value_cols:
                # Use count if no value column found
                product_counts = self.df[product_col].value_counts().head(top_n)
                fig = px.bar(
                    x=product_counts.values,
                    y=product_counts.index,
                    orientation='h',
                    title=f'Top {top_n} Products by Count',
                    labels={'x': 'Count', 'y': 'Product'}
                )
                return fig
            value_col = value_cols[0]

        # Aggregate by product
        product_performance = self.df.groupby(product_col)[value_col].sum().sort_values(ascending=False).head(top_n)

        # Create figure
        fig = px.bar(
            x=product_performance.values,
            y=product_performance.index,
            orientation='h',
            title=f'Top {top_n} Products by Sales',
            labels={'x': 'Total Sales ($)', 'y': 'Product'},
            color=product_performance.values,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray')
        )

        return fig

    def create_regional_analysis(
        self,
        region_col: str = None,
        value_col: str = None
    ) -> go.Figure:
        """
        Create regional analysis visualization.

        Args:
            region_col: Region column name (auto-detected if None)
            value_col: Value column name (auto-detected if None)

        Returns:
            Plotly figure
        """
        # Auto-detect columns
        if region_col is None:
            region_cols = [col for col in self.df.columns if 'region' in col.lower() or 'location' in col.lower()]
            if not region_cols:
                raise ValueError("No region column found")
            region_col = region_cols[0]

        if value_col is None:
            value_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
            if not value_cols:
                # Use count
                region_counts = self.df[region_col].value_counts()
                fig = px.pie(
                    values=region_counts.values,
                    names=region_counts.index,
                    title='Transactions by Region'
                )
                return fig
            value_col = value_cols[0]

        # Aggregate by region
        regional_performance = self.df.groupby(region_col)[value_col].agg(['sum', 'mean', 'count']).reset_index()

        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Total Sales by Region', 'Average Sale by Region'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Total sales
        fig.add_trace(
            go.Bar(
                x=regional_performance[region_col],
                y=regional_performance['sum'],
                name='Total Sales',
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        # Average sales
        fig.add_trace(
            go.Bar(
                x=regional_performance[region_col],
                y=regional_performance['mean'],
                name='Average Sale',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )

        fig.update_layout(
            title_text='Regional Performance Analysis',
            showlegend=False,
            plot_bgcolor='white'
        )

        return fig

    def create_customer_demographics(
        self,
        age_col: str = None,
        gender_col: str = None
    ) -> go.Figure:
        """
        Create customer demographics visualization.

        Args:
            age_col: Age column name (auto-detected if None)
            gender_col: Gender column name (auto-detected if None)

        Returns:
            Plotly figure
        """
        # Auto-detect columns
        if age_col is None:
            age_cols = [col for col in self.df.columns if 'age' in col.lower()]
            age_col = age_cols[0] if age_cols else None

        if gender_col is None:
            gender_cols = [col for col in self.df.columns if 'gender' in col.lower()]
            gender_col = gender_cols[0] if gender_cols else None

        if not age_col and not gender_col:
            raise ValueError("No demographic columns found")

        # Create subplots
        if age_col and gender_col:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Age Distribution', 'Gender Distribution'),
                specs=[[{"type": "histogram"}, {"type": "pie"}]]
            )

            # Age histogram
            fig.add_trace(
                go.Histogram(
                    x=self.df[age_col],
                    name='Age',
                    marker_color='skyblue',
                    nbinsx=20
                ),
                row=1, col=1
            )

            # Gender pie chart
            gender_counts = self.df[gender_col].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=gender_counts.index,
                    values=gender_counts.values,
                    name='Gender'
                ),
                row=1, col=2
            )

            fig.update_layout(title_text='Customer Demographics')

        elif age_col:
            # Just age
            fig = px.histogram(
                self.df,
                x=age_col,
                title='Customer Age Distribution',
                labels={age_col: 'Age', 'count': 'Number of Customers'},
                nbins=20
            )

        else:
            # Just gender
            gender_counts = self.df[gender_col].value_counts()
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title='Customer Gender Distribution'
            )

        return fig

    def create_correlation_heatmap(self, numeric_cols: List[str] = None) -> go.Figure:
        """
        Create correlation heatmap for numeric columns.

        Args:
            numeric_cols: List of numeric columns (auto-detected if None)

        Returns:
            Plotly figure
        """
        if numeric_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation")

        # Calculate correlation
        corr = self.df[numeric_cols].corr()

        # Create heatmap
        fig = px.imshow(
            corr,
            text_auto='.2f',
            aspect='auto',
            title='Correlation Heatmap',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1
        )

        return fig

    def create_time_series_decomposition(
        self,
        date_col: str = None,
        value_col: str = None
    ) -> go.Figure:
        """
        Create time series with moving average.

        Args:
            date_col: Date column name
            value_col: Value column name

        Returns:
            Plotly figure
        """
        # Auto-detect columns
        if date_col is None:
            # Look for columns with 'date' in the name (they're now strings after normalization)
            date_cols = [col for col in self.df.columns if 'date' in col.lower()]
            if len(date_cols) == 0:
                raise ValueError("No date column found")
            date_col = date_cols[0]

        if value_col is None:
            value_cols = [col for col in self.df.columns if 'sales' in col.lower()]
            if not value_cols:
                raise ValueError("No value column found")
            value_col = value_cols[0]

        # Sort by date and calculate moving average
        df_sorted = self.df.copy()
        # Convert string dates to datetime for sorting and plotting
        df_sorted['_temp_date'] = pd.to_datetime(df_sorted[date_col], errors='coerce')
        df_sorted = df_sorted.sort_values('_temp_date')
        df_sorted['MA_7'] = df_sorted[value_col].rolling(window=7).mean()
        df_sorted['MA_30'] = df_sorted[value_col].rolling(window=30).mean()

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_sorted['_temp_date'],
            y=df_sorted[value_col],
            mode='markers',
            name='Actual',
            marker=dict(size=4, color='lightblue')
        ))

        fig.add_trace(go.Scatter(
            x=df_sorted['_temp_date'],
            y=df_sorted['MA_7'],
            mode='lines',
            name='7-Day MA',
            line=dict(color='orange', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df_sorted['_temp_date'],
            y=df_sorted['MA_30'],
            mode='lines',
            name='30-Day MA',
            line=dict(color='red', width=2)
        ))

        fig.update_layout(
            title='Sales with Moving Averages',
            xaxis_title='Date',
            yaxis_title=value_col,
            hovermode='x unified',
            plot_bgcolor='white'
        )

        return fig

    def create_summary_dashboard(self) -> go.Figure:
        """
        Create a summary dashboard with key metrics.

        Returns:
            Plotly figure with multiple subplots
        """
        # Detect available columns
        sales_cols = [col for col in self.df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
        product_cols = [col for col in self.df.columns if 'product' in col.lower()]
        region_cols = [col for col in self.df.columns if 'region' in col.lower()]

        # Create a summary with available visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Key Metrics', 'Top Products', 'Regional Distribution', 'Summary Statistics'),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "table"}]
            ]
        )

        # Key metrics
        if sales_cols:
            total_sales = self.df[sales_cols[0]].sum()
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=total_sales,
                    title={"text": f"Total Sales"},
                    number={'prefix': "$", 'valueformat': ",.2f"}
                ),
                row=1, col=1
            )

        # Top products
        if product_cols and sales_cols:
            top_products = self.df.groupby(product_cols[0])[sales_cols[0]].sum().nlargest(5)
            fig.add_trace(
                go.Bar(x=top_products.values, y=top_products.index, orientation='h'),
                row=1, col=2
            )

        # Regional distribution
        if region_cols:
            region_counts = self.df[region_cols[0]].value_counts()
            fig.add_trace(
                go.Pie(labels=region_counts.index, values=region_counts.values),
                row=2, col=1
            )

        fig.update_layout(height=800, title_text="Business Intelligence Dashboard")

        return fig


def export_chart(fig: go.Figure, filename: str, format: str = 'png'):
    """
    Export chart to file.

    Args:
        fig: Plotly figure
        filename: Output filename
        format: Export format ('png', 'html', 'svg')
    """
    if format == 'html':
        fig.write_html(filename)
    elif format == 'png':
        fig.write_image(filename)
    elif format == 'svg':
        fig.write_image(filename, format='svg')
    else:
        raise ValueError(f"Unsupported format: {format}")
