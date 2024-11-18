import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
import yt_dlp
import os
import tempfile
import re
from dotenv import load_dotenv
import whisper
import torch
from utils.rag_utils import *
from utils.viz_utils import *

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube Content Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Whisper model
model = whisper.load_model("base", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def is_valid_youtube_url(url):
    """Validate YouTube URL"""
    if not url:
        return False
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    match = re.match(youtube_regex, url)
    return bool(match)

def download_youtube_audio(youtube_url, output_path):
    """Download YouTube audio using yt-dlp."""
    try:
        # Get the directory path from output_path
        output_dir = os.path.dirname(output_path)
        temp_filename = os.path.join(output_dir, "temp_audio.%(ext)s")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': temp_filename,  # Use the full path for temporary file
        }
    
        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        
        # Get the actual temp file path
        temp_file = os.path.join(output_dir, "temp_audio.wav")
        
        # If the temp file exists, move it to the desired output path
        if os.path.exists(temp_file):
            if os.path.exists(output_path):
                os.remove(output_path)  # Remove existing file if it exists
            os.rename(temp_file, output_path)
            return output_path
        else:
            st.error("Failed to download audio: Temporary file not found")
            return None

    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        # Clean up any temporary files if they exist
        temp_file = os.path.join(os.path.dirname(output_path), "temp_audio.wav")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Faster Whisper."""
    try:
        with st.spinner("🎯 Transcribing audio..."):
            result = model.transcribe(audio_path)
            return result["text"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None


def main():
    colored_header(
        label="🎥 YouTube Content Assistant",
        description="Ask questions about any YouTube video content",
        color_name="red-70"
    )
    
    st.markdown("""
    Welcome to YouTube Content Assistant! This app helps you:
    * 📝 Transcribe YouTube videos
    * 💡 Ask questions about the video content
    * 🤖 Get AI-powered responses
    """)

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        st.error("Please set your GROQ_API_KEY in the .env file")
        return

    # Initialize session state variables
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'transcription' not in st.session_state:
        st.session_state.transcription = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'keywords' not in st.session_state:
        st.session_state.keywords = None
    if 'sentiment' not in st.session_state:
        st.session_state.sentiment = None

    # Input for YouTube URL
    youtube_url = st.text_input(
        "🔗 Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste a valid YouTube video URL here"
    )
    
    if youtube_url:
        if not is_valid_youtube_url(youtube_url):
            st.error("Please enter a valid YouTube URL")
            return

        if st.button("Process Video"):
            try:
                # Create a temporary directory for audio files
                with tempfile.TemporaryDirectory() as temp_dir:
                    audio_path = os.path.join(temp_dir, "audio.wav")
                    
                    # Download and process
                    with st.spinner("🎵 Downloading audio..."):
                        audio_path = download_youtube_audio(youtube_url, audio_path)
                    if audio_path and os.path.exists(audio_path):
                        # Transcribe
                        
                        transcription = transcribe_audio(audio_path)
                        
                        if transcription:
                            st.session_state.transcription = transcription
                            
                            # Create embeddings and setup QA chain
                            with st.spinner("🧠 Setting up QA system..."):
                                try:
                                    vectorstore, texts = create_embeddings_from_text(transcription)
                                    st.session_state.qa_chain = setup_qa_chain(vectorstore)
                                    
                                    # Generate additional insights
                                    with st.spinner("✨ Generating insights..."):
                                        st.session_state.summary = generate_summary(transcription)
                                        st.session_state.keywords = extract_keywords(transcription)
                                        st.session_state.sentiment = analyze_sentiment(transcription)
                                    
                                    st.session_state.processed = True
                                    st.success("✅ Analysis complete! Scroll down to see insights.")
                                except Exception as e:
                                    st.error(f"Error setting up analysis: {str(e)}")
                                    return

                            # Create tabs for different views
                            tab1, tab2, tab3 = st.tabs(["📊 Summary & Insights", "📝 Transcription", "💭 Chat"])
                            
                            with tab1:
                                # Display summary
                                st.subheader("📝 Video Summary")
                                st.info(st.session_state.summary)
                                
                                # Display insights in columns
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Display keywords
                                    st.subheader("🏷️ Key Topics")
                                    for keyword in st.session_state.keywords:
                                        st.markdown(f"- {keyword}")
                                
                                with col2:
                                    # Display sentiment
                                    st.subheader("🎭 Content Sentiment")
                                    sentiment = st.session_state.sentiment
                                    
                                    # Create metrics for sentiment
                                    sentiment_emoji = {
                                        "positive": "😊",
                                        "negative": "😔",
                                        "neutral": "😐"
                                    }.get(sentiment["overall_sentiment"], "😐")
                                    
                                    st.metric(
                                        label=f"Overall Sentiment {sentiment_emoji}",
                                        value=sentiment["overall_sentiment"].upper(),
                                        delta=f"Confidence: {sentiment['confidence']:.2f}"
                                    )
                                    st.info(sentiment["brief_explanation"])
                            
                            with tab2:
                                st.subheader("📜 Full Transcription")
                                st.markdown(st.session_state.transcription)
                            
                            with tab3:
                                st.subheader("💭 Ask Questions")
                                # Question input
                                question = st.text_input(
                                    "❓ Ask a question about the video",
                                    placeholder="Type your question here...",
                                    key="question_input"
                                )
                                
                                if question:
                                    if st.session_state.qa_chain:
                                        with st.spinner("🤔 Thinking..."):
                                            try:
                                                response = get_response(st.session_state.qa_chain, question)
                                                st.session_state.conversation_history.append((question, response))
                                                
                                                # Display conversation
                                                st.markdown(f"**👤 You:** {question}")
                                                st.success(f"**🤖 Assistant:** {response['answer']}")
                                                
                                                # Show source context in expander
                                                with st.expander("🔍 View source context"):
                                                    for i, doc in enumerate(response['source_documents'], 1):
                                                        st.info(f"**Source {i}:**\n{doc.page_content}")
                                                
                                                # Display conversation history
                                                if len(st.session_state.conversation_history) > 1:
                                                    st.subheader("Previous Conversations")
                                                    for q, a in st.session_state.conversation_history[:-1]:
                                                        st.markdown(f"**👤 You:** {q}")
                                                        st.success(f"**🤖 Assistant:** {a['answer']}")
                                                        st.divider()
                                                        
                                            except Exception as e:
                                                st.error(f"Error generating response: {str(e)}")
                                    else:
                                        st.error("QA system is not ready. Please process the video first.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return

if __name__ == "__main__":
    main()
