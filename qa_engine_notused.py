import streamlit as st
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import utility functions
from utils import (
    check_password,
    check_required_files,
    setup_custom_css,
    load_evaluation_rules,
    check_api_connections,
    get_api_key
)

# Import knowledge base
from knowledge_base import KnowledgeBase

# Import chat QA module
from chat_qa import (
    render_chat_analysis_ui,
    process_chat_analysis,
    visualize_chat_results,
    analyze_chat_transcript
)

# Import batch processing module
from batch_processing import (
    render_batch_processing_ui,
    process_batch_analysis,
    visualize_batch_results
)

# Import enhanced chat processor
try:
    from enhanced_chat_processor import EnhancedChatProcessor
except ImportError:
    pass

# Import voice QA module (if available)
try:
    from voice_qa import (
        render_voice_analysis_ui,
        process_voice_analysis,
        check_voice_libraries,
        transcribe_audio,
        analyze_audio_quality,
        visualize_audio_quality
    )
    VOICE_MODULE_AVAILABLE = True
except ImportError:
    VOICE_MODULE_AVAILABLE = False
    
# Page configuration
st.set_page_config(
    page_title="CS QA Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup custom CSS
setup_custom_css()

def main():
    # First check password before showing any app content
    if not check_password():
        return
    
        # Update the title with an icon
        st.title("Customer Support QA Engine 🔍")
        st.caption("Beta version to test functionality")
    
    # Check for required files before proceeding
    required_files = {
        "evaluation_rules.json": "Text Chat Evaluation Rules",
        "scoring_system.json": "Scoring System",
        "QA_prompt.md": "Text QA Prompt Template",
        "qa_knowledge_base.json": "Knowledge Base"
    }
    
    # If voice module is available, add voice-specific files
    if VOICE_MODULE_AVAILABLE:
        required_files.update({
            "voice_evaluation_rules.json": "Voice Evaluation Rules",
            "voice_qa_prompt.md": "Voice QA Prompt Template"
        })
    
    # Check if required files are available
    check_required_files(required_files)
    
    # Create a container in the sidebar for status messages
    with st.sidebar:
        status_container = st.empty()
    
    # Debug mode toggle (hidden in sidebar expander for convenience)
    with st.sidebar.expander("Developer Options", expanded=False):
        debug_mode = st.checkbox("Debug Mode", value=False)
        max_workers = st.slider("Max Concurrent Workers", min_value=1, max_value=5, value=3, 
                              help="Maximum number of concurrent API calls for batch processing")
    
    try:
        # Initialize Knowledge Base
        kb = KnowledgeBase("qa_knowledge_base.json")
        
        # Load evaluation rules for text chat
        chat_rules = load_evaluation_rules("evaluation_rules.json", "scoring_system.json")
        
        # Load voice evaluation rules if available
        voice_rules = None
        if VOICE_MODULE_AVAILABLE:
            voice_rules = load_evaluation_rules("voice_evaluation_rules.json", "scoring_system.json")
        
        # Display success message in sidebar
        with st.sidebar:
            status_expander = st.expander("Status Messages", expanded=False)
            with status_expander:
                st.success("Loaded evaluation rules successfully")
                st.success(f"Loaded knowledge base with {len(kb.qa_pairs.get('qa_pairs', []))} Q&A pairs")
                if VOICE_MODULE_AVAILABLE and voice_rules:
                    st.success("Loaded voice evaluation rules successfully")
                elif VOICE_MODULE_AVAILABLE and not voice_rules:
                    st.warning("Voice module available but rules not loaded")
                elif not VOICE_MODULE_AVAILABLE:
                    st.info("Voice module not available")

        # Setup sidebar settings
        with st.sidebar:
            st.header("Settings")
            
            # Add model provider selection
            st.subheader("AI Model Settings")
            model_provider = st.radio(
                "Select Model Provider",
                ["Anthropic Claude", "OpenAI GPT"],
                index=0,
                key="model_provider"
            )
            
            # Convert friendly names to internal names
            provider_internal_name = "anthropic"
            if model_provider == "OpenAI GPT":
                provider_internal_name = "openai"
                model_name = "gpt-4o"
                st.info("Using GPT-4o")
            else:  # Anthropic
                provider_internal_name = "anthropic"
                model_name = "claude-3-7-sonnet-20250219"
                st.info("Using Claude 3.7 Sonnet")
            
            # API Key handling (dynamically based on selected provider)
            st.subheader(f"{model_provider} API Key")
            api_key = get_api_key(provider_internal_name)
            
            if not api_key:
                user_api_key = st.text_input(f"Enter your {model_provider} API key:", type="password")
                if user_api_key:
                    if provider_internal_name == "anthropic":
                        st.session_state.user_anthropic_api_key = user_api_key
                    elif provider_internal_name == "openai":
                        st.session_state.user_openai_api_key = user_api_key
                    st.rerun()
            else:
                st.success(f"✅ {model_provider} API key available")
            
            # Language settings
            st.subheader("Language Settings")
            auto_detect = st.checkbox("Auto-detect transcript language", value=True, key="auto_detect")
            
            if not auto_detect:
                transcript_language = st.selectbox(
                    "Transcript language",
                    ["English", "Chinese (Simplified)", "Chinese (Traditional)", "Vietnamese", "Thai", 
                    "Portuguese (Spain)", "Portuguese (Brazil)", "Spanish"],
                    key="transcript_language"
                )
                    
            output_language = st.radio(
                "Output language for analysis",
                ["English", "Native language of transcript"],
                key="output_language"
            )
            
            # Convert output_language to parameter for function
            target_language = "native" if output_language == "Native language of transcript" else "en"
            
            # About section
            st.markdown("---")
            st.caption("Customer Support QA Engine")
            st.caption("Pepperstone, Ltd.")

        # Check API connections and get the active provider
        provider_internal_name, model_name, model_provider = check_api_connections()
        
        # Stop if no API keys are available
        if not provider_internal_name:
            return
            
        # Main content area - only display if we have a valid API key
        st.markdown("Upload chat transcripts or voice calls for AI-powered quality assessment")
        
        # Add model info
        st.info(f"Using {model_provider} model: {model_name}")
        
        # Create tabs for all analysis modes
        available_tabs = ["Single Chat Analysis", "Batch Analysis"]
        if VOICE_MODULE_AVAILABLE:
            available_tabs.append("Voice Call Analysis")
            
        
        # Knowledge Base tab
        available_tabs.append("Knowledge Base")
        tabs = st.tabs(available_tabs)
        
        # Single Chat Analysis Tab
        with tabs[0]:
            # Render the chat analysis UI and get transcript
            chat_transcript, chat_prompt_path = render_chat_analysis_ui()
            
            # Analyze button
            if st.button("Analyze Transcript", key="analyze_chat"):
                if not chat_transcript:
                    st.warning("Please paste a transcript to analyze")
                else:
                    # Process transcript analysis
                    process_chat_analysis(
                        chat_transcript, 
                        chat_rules, 
                        kb,
                        target_language, 
                        chat_prompt_path,
                        provider_internal_name, 
                        model_name, 
                        debug_mode
                    )
            
            # Check if we have a stored result from previous analysis
            elif "current_result" in st.session_state and "current_transcript" in st.session_state:
                # Use the stored result
                result = st.session_state.current_result
                
                # Optional: Display a note that we're showing previous results
                st.info("Showing analysis for previously submitted transcript")
                
                if result:
                    # Visualize using the stored result
                    visualize_chat_results(result, chat_rules)
                    
                    # Show model info in result
                    st.caption(f"Analysis performed by {result.get('model_provider', 'Unknown').title()} {result.get('model_name', 'Unknown')}")
        
        # Batch Analysis Tab
        with tabs[1]:
            # Render batch processing UI
            selected_chats, processor = render_batch_processing_ui()
            
            if selected_chats:
                st.success(f"Found {len(selected_chats)} chats ready for analysis.")
                
                # Process batch button
                if st.button("Run Batch Analysis", key="process_batch"):
                    # Use batch processing
                    with st.spinner("Analyzing chat transcripts in batch..."):
                        # Show analysis settings
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"Analyzing {len(selected_chats)} chats")
                        with col2:
                            st.info(f"Using {model_provider} model: {model_name}")
                        
                        batch_results = process_batch_analysis(
                            selected_chats,
                            chat_rules,
                            kb,
                            target_language,
                            "QA_prompt.md",
                            provider_internal_name,
                            model_name,
                            max_workers
                        )
                        
                        # Store batch results in session state
                        st.session_state.batch_results = batch_results
                        st.session_state.batch_chats = selected_chats
                        
                        # Visualize batch results
                        if batch_results:
                            st.success(f"Successfully analyzed {len(batch_results)} chat transcripts")
                            visualize_batch_results(batch_results, chat_rules)
                        else:
                            st.error("No results were generated from batch analysis. Please check the logs for details.")
            
            # Check if we have stored batch results from previous analysis
            elif "batch_results" in st.session_state and "batch_chats" in st.session_state:
                # Use the stored results
                batch_results = st.session_state.batch_results
                
                # Display a note that we're showing previous results
                st.info("Showing analysis for previously submitted batch")
                
                if batch_results:
                    # Display previous analysis summary
                    st.success(f"Previous batch analysis: {len(batch_results)} chat transcripts")
                    # Visualize using the stored results
                    visualize_batch_results(batch_results, chat_rules)

        # Voice Call Analysis Tab (if available)
        if VOICE_MODULE_AVAILABLE and len(tabs) > 2:
            with tabs[2]:
                # Check if voice libraries are available
                if check_voice_libraries():
                    # Render voice analysis UI
                    audio_file, analyze_audio = render_voice_analysis_ui()
                    
                    if audio_file:
                        # Handle the audio transcription and quality analysis
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
                        else:
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
                            
                            # Offer to analyze the transcript
                            st.subheader("Analyze Transcript")
                            st.info("You can now analyze this transcript using the QA model.")
                            
                            do_analysis = st.button("Analyze Call Transcript", key="analyze_voice_transcript")
                            
                            if do_analysis:
                                status_text.text(f"Analyzing transcript with {model_provider}...")
                                with st.spinner(f"Analyzing transcript with {model_provider}..."):
                                    # Use the same analysis function for all models
                                    result = analyze_chat_transcript(
                                        transcript,
                                        voice_rules if voice_rules else chat_rules,
                                        kb,
                                        target_language,
                                        prompt_template_path="voice_qa_prompt.md" if os.path.exists("voice_qa_prompt.md") else "QA_prompt.md",
                                        model_provider=provider_internal_name,
                                        model_name=model_name
                                    )
                                    
                                    # Store result in session state
                                    st.session_state.current_result = result
                                    
                                    # Add language information to result
                                    if result:
                                        # Auto-detect language for transcript
                                        from utils import detect_language_cached
                                        text_sample = transcript[:200]
                                        lang_code, lang_name = detect_language_cached(text_sample, provider_internal_name)
                                        result["detected_language"] = lang_name
                                        
                                        # Display visualization
                                        try:
                                            visualize_chat_results(result, voice_rules if voice_rules else chat_rules)
                                            st.caption(f"Analysis performed by {model_provider}: {model_name}")
                                        except Exception as viz_error:
                                            st.error(f"Error visualizing results: {str(viz_error)}")
                                            st.json(result)  # Just show the raw result
                                
                            progress_bar.progress(100)
                            status_text.text("Voice call processing complete!")
        
        # Knowledge Base Tab
        if len(tabs) > 3:
            with tabs[3]:
                st.header("Knowledge Base Management")
                
                # Display existing KB entries
                st.subheader("Current Knowledge Base Entries")
                
                # Get all categories
                categories = kb.get_all_categories()
                
                # Allow filtering by category
                selected_category = st.selectbox(
                    "Filter by category",
                    ["All Categories"] + categories
                )
                
                # Display entries based on selection
                if selected_category == "All Categories":
                    qa_pairs = kb.qa_pairs.get("qa_pairs", [])
                else:
                    qa_pairs = kb.get_qa_pairs_by_category(selected_category)
                
                st.info(f"Showing {len(qa_pairs)} KB entries")
                
                # Display in an expander for each entry
                for i, qa_pair in enumerate(qa_pairs):
                    with st.expander(f"Q: {qa_pair['question'][:80]}...", expanded=False):
                        st.markdown(f"**Question:** {qa_pair['question']}")
                        st.markdown(f"**Answer:** {qa_pair['answer']}")
                        st.markdown(f"**Category:** {qa_pair['category']}")

                
    except Exception as e:
        # Error handling
        with st.sidebar:
            with st.expander("Error Details", expanded=True):
                st.error("An error occurred in the application:")
                if debug_mode:
                    st.exception(e)
                st.error(str(e))

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
