import streamlit as st
import json
import tempfile
import os
from datetime import datetime
import time
import io
from typing import Optional

# Import utilities from utils.py
from utils import (
    initialize_anthropic_client, 
    initialize_openai_client, 
    parse_json_response, 
    detect_language,
    detect_language_cached,
    load_prompt_template,
    load_evaluation_rules
)

# Import chat transcript analysis from chat_qa.py
from chat_qa import analyze_chat_transcript, visualize_chat_results

# Simplified voice libraries check
def check_voice_libraries():
    """
    Check if OpenAI API is available for voice transcription
    
    Returns:
        bool: True if OpenAI API is available, False otherwise
    """
    # We only need to ensure OpenAI client is available
    openai_client = initialize_openai_client()
    if not openai_client:
        st.error("""
        OpenAI API key is required for voice analysis.
        Please add your OpenAI API key in the sidebar settings.
        """)
        return False
    return True

@st.cache_data(ttl=3600, show_spinner=False)
def transcribe_audio_cached(audio_bytes):
    """
    Cached wrapper for audio transcription
    
    Args:
        audio_bytes: Audio file bytes
        
    Returns:
        Transcript text
    """
    # Create a hash of the audio bytes for caching
    import hashlib
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    
    # Call the actual transcription function
    return _transcribe_audio(audio_bytes)

def _transcribe_audio(audio_bytes):
    """
    Internal implementation of audio transcription using OpenAI's Whisper API
    """
    if not audio_bytes:
        return None
        
    try:
        # Get OpenAI client
        openai_client = initialize_openai_client()
        if not openai_client:
            st.error("OpenAI API key is required for Whisper transcription")
            return None
            
        # Create a temporary file for the audio
        import tempfile
        import os
        
        # Create temporary file with a .wav extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Open the temporary file and send it to OpenAI's API
            with open(temp_audio_path, "rb") as audio_file:
                # Use OpenAI's API directly - no need for ffmpeg or local processing
                transcription = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                
            # Extract the transcript text
            transcript = transcription.text
            
            # Clean up the temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as cleanup_error:
                st.warning(f"Could not clean up temporary file: {str(cleanup_error)}")
                
            return transcript
            
        except Exception as api_error:
            st.error(f"Error calling OpenAI Whisper API: {str(api_error)}")
            # Clean up temp file in case of error
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            return None
            
    except Exception as e:
        import traceback
        st.error(f"Error transcribing audio: {str(e)}")
        st.error(traceback.format_exc())
        return None

def transcribe_audio(audio_file):
    """
    Transcribe audio file using OpenAI's Whisper API
    
    Args:
        audio_file: Uploaded audio file (BytesIO)
        
    Returns:
        Transcript text
    """
    # Get the audio bytes for caching
    audio_bytes = audio_file.getvalue()
    
    # Use the cached version
    return transcribe_audio_cached(audio_bytes)

@st.cache_data(ttl=3600)
def analyze_audio_quality_cached(audio_bytes):
    """
    Cached wrapper for audio quality analysis
    
    Args:
        audio_bytes: Audio file bytes
        
    Returns:
        Dictionary of audio quality metrics
    """
    # Create a hash of the audio bytes for caching
    import hashlib
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    
    # Call the actual analysis function
    return _analyze_audio_quality(audio_bytes)

def _analyze_audio_quality(audio_bytes):
    """
    Simplified audio quality analysis that doesn't require ffmpeg
    """
    try:
        # Basic file size and duration estimates
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        
        # Estimate duration based on typical WAV file sizes
        # (Very rough approximation: ~10MB per minute for 16-bit stereo 44.1kHz)
        estimated_duration = file_size_mb * 6  # seconds
        
        # Basic quality score based on file size
        quality_score = 80  # Default score
        quality_issues = []
        
        # Check file size
        if file_size_mb < 0.2:  # Less than 200KB
            quality_score -= 30
            quality_issues.append("Audio file is very small, may have low quality")
        
        # Check estimated duration
        if estimated_duration < 5:  # Less than 5 seconds
            quality_score -= 20
            quality_issues.append("Audio clip is very short")
        
        # Ensure score is between
        quality_score = max(0, min(100, quality_score))
        
        return {
            "duration_seconds": estimated_duration,
            "file_size_mb": file_size_mb,
            "sample_rate": "Unknown (using OpenAI Whisper API)",
            "channels": "Unknown (using OpenAI Whisper API)",
            "silence_percentage": "Unknown (using OpenAI Whisper API)",
            "quality_score": quality_score,
            "quality_issues": quality_issues
        }
        
    except Exception as e:
        st.error(f"Error analyzing audio quality: {str(e)}")
        return None

def analyze_audio_quality(audio_file):
    """
    Analyze audio quality metrics from the uploaded file
    
    Args:
        audio_file: Uploaded audio file
        
    Returns:
        Dictionary of audio quality metrics
    """
    # Get the audio bytes for caching
    audio_bytes = audio_file.getvalue()
    
    # Use the cached version
    return analyze_audio_quality_cached(audio_bytes)

def analyze_voice_transcript(transcript, rules, target_language="en", prompt_template_path="voice_qa_prompt.md", 
                            model_provider="anthropic", model_name=None):
    """
    Analyze a voice call transcript using the specified model and prompt template.
    This is a wrapper around analyze_chat_transcript with voice-specific configuration.
    
    Args:
        transcript (str): The voice call transcript text
        rules (dict): Voice evaluation rules dictionary
        target_language (str): Target language for analysis response ('en' or 'native')
        prompt_template_path (str): Path to the voice prompt template file
        model_provider (str): AI provider ("anthropic" or "openai")
        model_name (str): Model name to use
        
    Returns:
        dict: Analysis results or None if analysis failed
    """
    # Use the standard analysis function
    return analyze_chat_transcript(
        transcript, 
        rules, 
        target_language,
        prompt_template_path=prompt_template_path,
        model_provider=model_provider,
        model_name=model_name
    )

def visualize_audio_quality(audio_quality):
    """
    Visualize audio quality analysis results
    
    Args:
        audio_quality (dict): Audio quality analysis results
    """
    if not audio_quality:
        return
        
    # Create audio quality score box
    quality_score = audio_quality["quality_score"]
    
    # Determine quality level based on score
    quality_level = "Poor"
    color_class = "score-box-poor"
    
    if quality_score >= 90:
        quality_level = "Excellent" 
        color_class = "score-box-excellent"
    elif quality_score >= 75:
        quality_level = "Good"
        color_class = "score-box-good"
    elif quality_score >= 50:
        quality_level = "Needs Improvement"
        color_class = "score-box-needs-improvement"
    
    # Display quality score
    st.markdown(
        f"<div class='score-box {color_class}'>"
        f"<h2 style='color:#333333;'>Audio Quality Score: {quality_score:.1f}</h2>"
        f"<p style='font-size:18px; color:#333333;'>Quality Level: {quality_level}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Display technical details in an expander
    with st.expander("Technical Details", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if "duration_seconds" in audio_quality:
                st.metric("Estimated Duration", f"{audio_quality['duration_seconds']:.1f} seconds")
            if "file_size_mb" in audio_quality:
                st.metric("File Size", f"{audio_quality['file_size_mb']:.2f} MB")
    
    # Display quality issues if any
    if audio_quality["quality_issues"]:
        st.warning("Quality Issues Detected:")
        for issue in audio_quality["quality_issues"]:
            st.markdown(f"- {issue}")
    else:
        st.success("No significant audio quality issues detected.")

def render_voice_analysis_ui():
    """
    Renders the voice call analysis UI components
    
    Returns:
        tuple: Audio file, analyze_audio flag
    """
    if not check_voice_libraries():
        st.warning("Voice analysis features require an OpenAI API key.")
        return None, None
    
    # Add clear information about using OpenAI Whisper
    st.info("📢 Voice transcription is powered by OpenAI's Whisper API")
    
    # Audio file upload
    audio_file = st.file_uploader("Upload a voice call recording", type=["mp3", "wav", "ogg", "m4a"])
    
    if audio_file:
        st.audio(audio_file)
        
        analyze_audio = st.checkbox(
            "Basic audio quality check",
            value=True,
            help="Perform a basic quality assessment of the audio file"
        )
            
        return audio_file, analyze_audio
    
    return None, None

def process_voice_analysis(audio_file, analyze_audio, rules, 
                          target_language="en", prompt_path="voice_qa_prompt.md",
                          provider_internal_name="anthropic", model_name=None):
    """
    Process a voice call analysis request
    
    Args:
        audio_file: The uploaded audio file
        analyze_audio: Whether to analyze audio quality
        rules: Evaluation rules dictionary
        target_language: Target language for analysis
        prompt_path: Path to the prompt template
        provider_internal_name: Provider name
        model_name: Model name
        
    Returns:
        tuple: (transcript, audio_quality, analysis_result)
    """
    if not audio_file:
        return None, None, None
        
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Transcribe audio
    status_text.text("Transcribing audio using OpenAI Whisper...")
    progress_bar.progress(10)
    
    transcript = transcribe_audio(audio_file)
    
    if not transcript:
        status_text.text("Transcription failed!")
        progress_bar.progress(100)
        st.error("Failed to transcribe the audio. Please check the file format and try again.")
        return None, None, None
    
    progress_bar.progress(40)
    
    # Step 2: Analyze audio quality if requested
    audio_quality = None
    if analyze_audio:
        status_text.text("Analyzing audio quality...")
        audio_quality = analyze_audio_quality(audio_file)
        
        if audio_quality:
            progress_bar.progress(60)
        else:
            st.warning("Audio quality analysis failed, continuing with transcription only.")
    
    progress_bar.progress(70)
    
    # Step 3: Display transcript
    status_text.text("Processing transcript...")
    
    st.subheader("Transcript")
    st.caption("Transcribed by OpenAI Whisper API")
    st.text_area("Voice Call Transcript", transcript, height=300)
    
    # Store transcript in session state
    st.session_state.current_transcript = transcript
    
    # Step 4: Show audio quality analysis results if available
    if audio_quality:
        st.subheader("Audio Quality Analysis")
        visualize_audio_quality(audio_quality)
    
    progress_bar.progress(90)
    
    # Offer to analyze the transcript with the QA model
    st.subheader("Analyze Transcript")
    st.info("You can now analyze this transcript using the QA model.")
    
    do_analysis = st.button("Analyze Call Transcript", key="analyze_voice_transcript")
    
    analysis_result = None
    if do_analysis:
        # Use the voice transcript analysis function
        status_text.text("Analyzing transcript...")
        model_provider_display = "Anthropic Claude" if provider_internal_name == "anthropic" else "OpenAI GPT"
        
        with st.spinner(f"Analyzing transcript with {model_provider_display}..."):
            analysis_result = analyze_voice_transcript(
                transcript, 
                rules, 
                target_language=target_language,
                prompt_template_path=prompt_path,
                model_provider=provider_internal_name,
                model_name=model_name
            )
            
            # Store result in session state
            st.session_state.current_result = analysis_result
            
            # Add language information to result
            if analysis_result:
                # Auto-detect language for transcript
                text_sample = transcript[:200]
                lang_code, lang_name = detect_language_cached(text_sample, provider_internal_name)
                analysis_result["detected_language"] = lang_name
                
                # Display visualization
                try:
                    visualize_chat_results(analysis_result, rules)
                    st.caption(f"Analysis performed by {model_provider_display} {model_name}")
                except Exception as viz_error:
                    st.error(f"Error visualizing results: {str(viz_error)}")
                    st.json(analysis_result)  # Just show the raw result
                    
    progress_bar.progress(100)
    status_text.text("Voice call processing complete!")
    
    return transcript, audio_quality, analysis_result