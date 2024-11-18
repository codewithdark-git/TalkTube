# 🎥 YouTube Content Assistant

A powerful Streamlit application that allows users to analyze and interact with YouTube video content through natural language questions.

## 🌟 Features

- **YouTube Video Processing**: Input any valid YouTube URL to analyze its content
- **Audio Transcription**: Automatically transcribes video content using Whisper AI
- **Interactive Q&A**: Ask questions about the video content using advanced RAG (Retrieval-Augmented Generation)
- **GPU Acceleration**: Utilizes CUDA for faster processing when available
- **User-Friendly Interface**: Clean and intuitive Streamlit interface

## 🚀 Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd TalkTube
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with necessary API keys and configurations.

## 📦 Dependencies

- streamlit
- streamlit-extras
- yt-dlp
- whisper
- torch
- python-dotenv
- (other dependencies as specified in requirements.txt)

## 🎮 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter a YouTube URL in the input field
3. Wait for the video to be processed and transcribed
4. Ask questions about the video content in the chat interface

## 💡 How It Works

1. **Video Processing**: The app downloads the audio from YouTube videos using yt-dlp
2. **Transcription**: Uses OpenAI's Whisper model to transcribe the audio content
3. **Question Answering**: Implements RAG (Retrieval-Augmented Generation) to provide accurate answers based on the video content

## 🛠️ Technical Details

- Built with Streamlit for the web interface
- Uses Whisper AI for accurate speech-to-text transcription
- Implements advanced RAG techniques for question answering
- Supports both CPU and GPU processing
- Handles various YouTube URL formats

## ⚠️ Notes

- Processing time depends on video length and available computing resources
- GPU acceleration significantly improves transcription speed
- Internet connection required for YouTube video download and processing
