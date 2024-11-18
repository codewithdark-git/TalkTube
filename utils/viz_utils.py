import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import numpy as np
from typing import List, Dict
import pandas as pd

def create_sentiment_timeline(segments: List[Dict]) -> go.Figure:
    """Create a sentiment timeline visualization"""
    df = pd.DataFrame(segments)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='blue'),
        hovertemplate='Time: %{x}<br>Sentiment: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Sentiment Timeline',
        xaxis_title='Time',
        yaxis_title='Sentiment Score',
        hovermode='x unified',
        showlegend=False
    )
    
    return fig

def generate_word_cloud(text: str, width: int = 800, height: int = 400) -> WordCloud:
    """Generate a word cloud from text"""
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        min_font_size=10,
        max_font_size=50
    ).generate(text)
    
    return wordcloud

def create_topic_distribution(topics: Dict[str, float]) -> go.Figure:
    """Create a pie chart of topic distribution"""
    fig = go.Figure(data=[go.Pie(
        labels=list(topics.keys()),
        values=list(topics.values()),
        hole=.3
    )])
    
    fig.update_layout(
        title='Topic Distribution',
        showlegend=True
    )
    
    return fig

def create_keyword_frequency_chart(keywords: Dict[str, int], top_n: int = 10) -> go.Figure:
    """Create a bar chart of keyword frequencies"""
    # Sort keywords by frequency and take top N
    sorted_keywords = dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    fig = go.Figure(data=[go.Bar(
        x=list(sorted_keywords.keys()),
        y=list(sorted_keywords.values()),
        text=list(sorted_keywords.values()),
        textposition='auto',
    )])
    
    fig.update_layout(
        title=f'Top {top_n} Keywords',
        xaxis_title='Keywords',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return fig

def create_speaker_timeline(segments: List[Dict]) -> go.Figure:
    """Create a timeline visualization of different speakers"""
    df = pd.DataFrame(segments)
    
    fig = px.timeline(
        df,
        x_start='start_time',
        x_end='end_time',
        y='speaker',
        color='speaker',
        hover_data=['text']
    )
    
    fig.update_layout(
        title='Speaker Timeline',
        xaxis_title='Time',
        yaxis_title='Speaker',
        showlegend=True
    )
    
    return fig
