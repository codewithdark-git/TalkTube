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

    except ConnectionError:
        st.error(f"Error downloading audio: Check Your Internet connection")
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


@st.cache_data
def process_video_data(youtube_url):
    """Cache the video processing results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = os.path.join(temp_dir, "audio.wav")
        audio_path = download_youtube_audio(youtube_url, audio_path)
        if audio_path and os.path.exists(audio_path):
            return transcribe_audio(audio_path)
    return None

def get_video_details(url):
    """Get video details using yt-dlp."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown Title'),
                'thumbnail': info.get('thumbnail', None),
                'channel': info.get('uploader', 'Unknown Channel'),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', ''),
                'description': info.get('description', ''),
                'like_count': info.get('like_count', 0)
            }
    except Exception as e:
        st.error(f"Error getting video details: {str(e)}")
        return None

def format_number(num):
    """Format large numbers in a readable way."""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(num)

def format_date(date_str):
    """Format date string to readable format."""
    if len(date_str) == 8:
        return f"{date_str[6:8]}-{date_str[4:6]}-{date_str[0:4]}"
    return date_str

def display_video_details(details):
    """Display video thumbnail and details in a beautiful format."""
    if details and details['thumbnail']:
        # Create a container for video details
        with st.container():
            # Create columns with better proportions
            col1, col2 = st.columns([1, 1.5], gap="large")
            
            with col1:
                # Add spacing and display image
                st.write("")
                st.write("")
                st.image(
                    details['thumbnail'],
                    width=350,
                    caption="🎬 Preview"
                )
                st.write("")
            
            with col2:
                # Title and channel section
                st.markdown(f"## 🎥 {details['title']}")
                st.markdown(f"### 👤 {details['channel']}")
                
                # Video stats in columns
                stat_cols = st.columns(3)
                
                with stat_cols[0]:
                    duration = f"{details['duration'] // 60}:{(details['duration'] % 60):02d}"
                    st.metric("⏱️ Duration", duration)
                
                with stat_cols[1]:
                    if details.get('view_count'):
                        st.metric("👀 Views", format_number(details['view_count']))
                
                with stat_cols[2]:
                    if details.get('like_count'):
                        st.metric("👍 Likes", format_number(details['like_count']))
                
                # Upload date
                if details.get('upload_date'):
                    st.markdown(f"**📅 Upload Date:** {format_date(details['upload_date'])}")


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

    # Create a form for URL input
    with st.form(key='url_form'):
        youtube_url = st.text_input(
            "🔗 Enter YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a valid YouTube video URL here"
        )
        submit_button = st.form_submit_button("Process Video")

    if submit_button and youtube_url:
        if not is_valid_youtube_url(youtube_url):
            st.error("Please enter a valid YouTube URL")
            return

        try:
            # First get and display video details
            with st.spinner("📺 Loading video details..."):
                video_details = get_video_details(youtube_url)
                if video_details:
                    display_video_details(video_details)
                    st.divider()

            # Process video with caching
            transcription = process_video_data(youtube_url)
            
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

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            return

    # Only show tabs if processing is complete
    if st.session_state.processed:
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
                st.markdown(", ".join(f"`{keyword}`" for keyword in st.session_state.keywords))
            
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
            # Create a form for questions
            with st.form(key='question_form'):
                question = st.text_input(
                    "❓ Ask a question about the video",
                    placeholder="Type your question here...",
                    key="question_input"
                )
                ask_button = st.form_submit_button("Ask Question")
            
            if ask_button and question:
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

if __name__ == "__main__":
    main()
