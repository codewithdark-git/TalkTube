# YouTube Content Assistant App

An advanced YouTube video analysis tool that provides transcription, semantic search, and interactive Q&A capabilities.

## Features

- YouTube video URL input and processing
- Audio extraction and transcription using Whisper
- Semantic search and RAG-based Q&A system
- Video summarization and keyword search
- Content insights and visualization
- Interactive video playback with synchronized transcription

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter a YouTube URL or upload a video file
2. Wait for the processing to complete
3. Explore the various analysis features:
   - View transcription
   - Ask questions about the content
   - Search for keywords
   - Generate summaries
   - Analyze insights

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- OpenAI API key
