# Add imports at the top
import concurrent.futures
import streamlit as st
import time
import re
import os
import io
import hashlib
import datetime
from typing import List, Dict, Any, Optional

# Import the enhanced chat processor
from enhanced_chat_processor import EnhancedChatProcessor

# Try to import file handling libraries
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

# Initialize session state for batch processing
if 'batch_mode' not in st.session_state:
    st.session_state.batch_mode = True

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

if 'selected_chats' not in st.session_state:
    st.session_state.selected_chats = []

if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Batch Analysis"

# Function to analyze a single chat with error handling
# Function to analyze a single chat with improved error handling
def analyze_single_chat(chat, rules, target_language, prompt_path, provider_internal_name, model_name):
    """
    Analyze a single chat with improved error handling and retries
    
    Args:
    chat: Chat dictionary containing processed_content
    rules: Evaluation rules
    target_language: Target language for analysis
    prompt_path: Path to prompt template
    provider_internal_name: Provider name (anthropic, openai)
    model_name: Model name
    
    Returns:
    Analysis result or None on failure
    """
    max_retries = 2
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            # Import at function level to avoid circular imports
            from chat_qa import analyze_chat_transcript
            from utils import detect_language_cached
            
            # Check if chat content is valid
            if not chat.get('processed_content') or len(chat['processed_content'].strip()) < 50:
                return {
                    "success": False,
                    "error": "Chat content too short or empty",
                    "chat_id": chat['id']
                }
            
            # Analyze the chat
            result = analyze_chat_transcript(
                chat['processed_content'], 
                rules, 
                target_language,
                prompt_template_path=prompt_path,
                model_provider=provider_internal_name,
                model_name=model_name
            )
            
            if result:
                # Auto-detect language - use cached version
                text_sample = chat['processed_content'][:200]
                lang_code, lang_name = detect_language_cached(text_sample, provider_internal_name)
                result["detected_language"] = lang_name
                result["chat_id"] = chat['id']
                
                # Include chat metadata but limit content size to avoid memory issues
                result["content_preview"] = chat['content'][:500] + "..." if len(chat['content']) > 500 else chat['content']
                
                # Validate result structure before returning
                for param in rules["parameters"]:
                    param_name = param["name"]
                    if param_name not in result:
                        result[param_name] = {
                            "score": 50,  # Default score
                            "explanation": "No analysis available for this parameter",
                            "example": "N/A",
                            "suggestion": "Please review manually"
                        }
                
                return {
                    "success": True,
                    "result": result,
                    "chat_id": chat['id']
                }
            else:
                if retry_count < max_retries:
                    retry_count += 1
                    time.sleep(3)  # Wait before retry
                    continue
                return {
                    "success": False,
                    "error": "Analysis failed after multiple attempts",
                    "chat_id": chat['id']
                }
        
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "too many requests" in error_msg.lower():
                if retry_count < max_retries:
                    retry_count += 1
                    # Exponential backoff
                    wait_time = 5 * (2 ** retry_count)
                    time.sleep(wait_time)
                    continue
            
            return {
                "success": False,
                "error": error_msg,
                "chat_id": chat['id']
            }
        
        # If we made it here without returning, increment retry counter
        retry_count += 1
        


# Update the process_batch_analysis function with improved error logging
def process_batch_analysis(selected_chats, rules, target_language, prompt_path, provider_internal_name, model_name, max_workers=2):
    """Process batch analysis for selected chats with improved concurrent processing"""
    
    st.header("Batch Analysis Results")
    
    # Show processing UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get batch size
    batch_size = len(selected_chats)
    if batch_size == 0:
        st.warning("No chats selected for analysis.")
        return []
    
    # Create debug log container
    debug_log = st.expander("Debug Log", expanded=False)
    with debug_log:
        st.write(f"Starting batch analysis with {batch_size} chats")
        st.write(f"Provider: {provider_internal_name}, Model: {model_name}")
    
    # Create placeholders for status updates
    status_text.text(f"Preparing to analyze {batch_size} chats...")
    
    # Create a list to collect results
    results = []
    failures = []
    
    # Create rate limiter to avoid hitting API limits
    # Rate limiter - reduced to 2 requests per minute per worker to be safe
    calls_per_minute = 2 * max_workers
    min_delay = 60.0 / calls_per_minute  # Minimum time between API calls in seconds
    last_call_time = time.time() - min_delay  # Initialize to allow immediate first call
    
    # Process one chat first as a test
    try:
        test_chat = selected_chats[0]
        with debug_log:
            st.write(f"Testing analysis with first chat: {test_chat['id']}")
            st.write(f"Chat content length: {len(test_chat.get('processed_content', ''))}")
        
        # Mark call time before making API request
        last_call_time = time.time()
        
        # Analyze chat with error handling
        test_data = analyze_single_chat(
            test_chat, rules, target_language, prompt_path, provider_internal_name, model_name
        )
        
        with debug_log:
            st.write(f"Test analysis result: {'Success' if test_data.get('success') else 'Failed'}")
            if not test_data.get('success'):
                st.write(f"Error: {test_data.get('error', 'Unknown error')}")
    
    except Exception as e:
        with debug_log:
            st.write(f"Error during test analysis: {str(e)}")
            import traceback
            st.write(traceback.format_exc())
        st.error(f"Error during test analysis: {str(e)}")
        return []  # Return empty results if test fails
    
    # Process sequentially for better reliability with smaller batches
    if batch_size <= 3:
        with debug_log:
            st.write("Using sequential processing for small batch")
        
        st.info("Small batch detected - using sequential processing for reliability")
        
        for i, chat in enumerate(selected_chats):
            # Rate limiting
            elapsed = time.time() - last_call_time
            if elapsed < min_delay:
                time.sleep(min_delay - elapsed)
                
            chat_id = chat['id']
            status_text.text(f"Analyzing chat {i+1}/{batch_size}: {chat_id}")
            
            try:
                with debug_log:
                    st.write(f"Processing chat {i+1}: {chat_id}")
                
                # Mark call time before making API request
                last_call_time = time.time()
                
                # Analyze chat with error handling
                data = analyze_single_chat(
                    chat, rules, target_language, prompt_path, provider_internal_name, model_name
                )
                
                if data and data.get("success"):
                    with debug_log:
                        st.write(f"Successfully analyzed chat {i+1}")
                    
                    results.append(data["result"])
                    st.success(f"Successfully analyzed chat {i+1}: {chat_id}")
                else:
                    with debug_log:
                        st.write(f"Failed to analyze chat {i+1}: {data.get('error', 'Unknown error')}")
                    
                    failures.append({
                        "chat_id": chat_id,
                        "index": i + 1,
                        "error": data.get('error', 'Unknown error')
                    })
                    st.error(f"Failed to analyze chat {i+1}: {chat_id} - {data.get('error', 'Unknown error')}")
            
            except Exception as e:
                with debug_log:
                    st.write(f"Exception analyzing chat {i+1}: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
                
                failures.append({
                    "chat_id": chat_id,
                    "index": i + 1,
                    "error": str(e)
                })
                st.error(f"Error analyzing chat {i+1}: {chat_id} - {str(e)}")
            
            # Update progress
            progress_value = (i + 1) / batch_size
            progress_bar.progress(progress_value)
    
    # Use concurrent processing for larger batches
    else:
        # Add some backoff for larger batches - reduce max_workers if batch is large
        if batch_size > 10 and max_workers > 2:
            max_workers = 2
            st.info(f"Large batch detected - reducing concurrency to {max_workers} workers")
        
        with debug_log:
            st.write(f"Using concurrent processing with {max_workers} workers for batch of {batch_size} chats")
        
        # Process in batches with progress tracking
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a list of futures for better tracking
            futures = []
            for i, chat in enumerate(selected_chats):
                # Add a small delay between submissions to avoid overwhelming the API
                if i > 0:
                    time.sleep(min_delay / max_workers)
                
                with debug_log:
                    st.write(f"Submitting chat {i+1} for analysis: {chat['id']}")
                
                future = executor.submit(
                    analyze_single_chat, 
                    chat, 
                    rules, 
                    target_language, 
                    prompt_path, 
                    provider_internal_name, 
                    model_name
                )
                futures.append((future, i, chat['id']))
            
            # Process results as they complete
            completed = 0
            for future, chat_index, chat_id in futures:
                try:
                    # Wait for the future with a timeout
                    with debug_log:
                        st.write(f"Waiting for result of chat {chat_index+1}: {chat_id}")
                    
                    data = future.result(timeout=120)  # 2-minute timeout
                    
                    if data and data.get("success"):
                        with debug_log:
                            st.write(f"Successfully analyzed chat {chat_index+1}")
                        
                        results.append(data["result"])
                        st.success(f"Successfully analyzed chat {chat_index+1}: {chat_id}")
                    else:
                        with debug_log:
                            st.write(f"Failed to analyze chat {chat_index+1}: {data.get('error', 'Unknown error')}")
                        
                        failures.append({
                            "chat_id": chat_id,
                            "index": chat_index + 1,
                            "error": data.get('error', 'Unknown error') if data else "No data returned"
                        })
                        st.error(f"Failed to analyze chat {chat_index+1}: {chat_id} - {data.get('error', 'Unknown error') if data else 'No data returned'}")
                    
                except concurrent.futures.TimeoutError:
                    with debug_log:
                        st.write(f"Timeout analyzing chat {chat_index+1}: {chat_id}")
                    
                    failures.append({
                        "chat_id": chat_id,
                        "index": chat_index + 1,
                        "error": "Analysis timed out after 2 minutes"
                    })
                    st.error(f"Timeout analyzing chat {chat_index+1}: {chat_id}")
                    
                except Exception as e:
                    with debug_log:
                        st.write(f"Exception analyzing chat {chat_index+1}: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())
                    
                    failures.append({
                        "chat_id": chat_id,
                        "index": chat_index + 1,
                        "error": str(e)
                    })
                    st.error(f"Error analyzing chat {chat_index+1}: {chat_id} - {str(e)}")
                
                # Update progress
                completed += 1
                progress_value = completed / batch_size
                progress_bar.progress(progress_value)
                status_text.text(f"Processed {completed} of {batch_size} chats...")
                
                # Small delay to prevent UI from freezing
                time.sleep(0.1)

    with debug_log:
        st.write(f"Batch processing complete. Successful: {len(results)}, Failed: {len(failures)}")
    
    # Finalize progress
    progress_bar.progress(1.0)
    
    # Summary message
    if len(results) == batch_size:
        status_text.text(f"Batch analysis complete! All {batch_size} chats successfully analyzed.")
    else:
        status_text.text(f"Batch analysis complete with {len(failures)} failures. Successfully analyzed {len(results)} of {batch_size} chats.")
    
    # Show failures in an expander
    if failures:
        with st.expander("View Analysis Failures"):
            st.write("The following chats could not be analyzed:")
            for failure in failures:
                st.write(f"• Chat {failure['index']} (ID: {failure['chat_id']}): {failure['error']}")
    
    return results

def render_batch_processing_ui():
    """Render a simplified UI for batch processing with reduced page resets"""
    st.header("Batch Chat Analysis")
    
    # Set batch mode
    st.session_state.batch_mode = True
    
    # Initialize session state variables if they don't exist
    if 'file_hashes' not in st.session_state:
        st.session_state.file_hashes = []
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload chat transcript files (CSV, DOCX, PDF, TXT)",
        type=['csv', 'docx', 'pdf', 'txt'],
        accept_multiple_files=True,
        key="batch_file_uploader"
    )
    
    # If no files are uploaded, show message and check for cached results
    if not uploaded_files:
        st.info("Please upload one or more chat transcript files to begin batch analysis.")
        
        # Check if we have stored chats in session state
        if 'selected_chats' in st.session_state and 'processor' in st.session_state:
            return st.session_state.selected_chats, st.session_state.processor
        return None, None
    
    # Check if the uploaded files are the same as previously processed
    current_hashes = []
    for uploaded_file in uploaded_files:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        current_hashes.append(file_hash)
    
    # If hashes match, return cached results
    if current_hashes == st.session_state.file_hashes and 'selected_chats' in st.session_state:
        st.success(f"Using cached results: {len(st.session_state.selected_chats)} chats loaded")
        return st.session_state.selected_chats, st.session_state.processor
    
    # Store current file hashes
    st.session_state.file_hashes = current_hashes
    
    # We have new files or different files, process them
    st.session_state.processed_files = {}  # Reset processed files
    
    # Initialize the enhanced chat processor
    processor = EnhancedChatProcessor()
    
    # Process all uploaded files
    all_chats = []
    
    # Add progress indicators
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    total_files = len(uploaded_files)
    for i, uploaded_file in enumerate(uploaded_files):
        progress_text.text(f"Processing file {i+1}/{total_files}: {uploaded_file.name}")
        progress_bar.progress((i) / total_files)
        
        # Generate a unique key for this file
        file_content_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()[:8]
        file_key = f"{uploaded_file.name}_{file_content_hash}"
        
        try:
            # Process the file
            with st.spinner(f"Extracting chats from {uploaded_file.name}..."):
                chats = processor.extract_chats_from_file(uploaded_file)
            
            if chats:
                # Store in session state for caching
                st.session_state.processed_files[file_key] = chats
                all_chats.extend(chats)
            else:
                st.warning(f"No valid chats found in {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    # Complete the progress bar
    progress_bar.progress(1.0)
    
    # Display summary of processed files
    if all_chats:
        st.success(f"Successfully extracted {len(all_chats)} chat conversations")
        
        # Store the all chats and processor in session state
        st.session_state.selected_chats = all_chats
        st.session_state.processor = processor
        
        # Return the all chats and processor
        return all_chats, processor
    else:
        st.warning("No chats were extracted from the uploaded files.")
        return None, None

def visualize_batch_results(results, rules):
    """Visualize batch analysis results with simplified reporting"""
    import json
    from datetime import datetime
    import io
    import hashlib
    
    if not results:
        st.warning("No analysis results to display.")
        return
    
    # Generate a timestamp if not already in session state for this batch
    # Create a unique identifier for this batch based on results content
    if 'current_batch_id' not in st.session_state:
        # Create a hash of the first result to identify this batch
        batch_hash = hashlib.md5(str(results[0]).encode()).hexdigest()[:8]
        st.session_state.current_batch_id = batch_hash
        st.session_state.batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    timestamp = st.session_state.batch_timestamp
    batch_id = st.session_state.current_batch_id
    
    # Store results in session state if not already there
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = results
    
    # Use session state results
    results = st.session_state.batch_results
    
    # Create summary statistics
    overall_scores = [result.get('weighted_overall_score', 0) for result in results]
    avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    
    # Determine quality level
    quality_level = "Unknown"
    if "scoring_system" in rules and "quality_levels" in rules["scoring_system"]:
        for level in rules["scoring_system"]["quality_levels"]:
            if level["range"]["min"] <= avg_score <= level["range"]["max"]:
                quality_level = level["name"]
                break
    
    # Set color class based on quality level
    color_class = "score-box-needs-improvement"  # Default
    if "excellent" in quality_level.lower():
        color_class = "score-box-excellent"
    elif "good" in quality_level.lower():
        color_class = "score-box-good"
    elif "poor" in quality_level.lower():
        color_class = "score-box-poor"
    
    # Display batch summary
    st.markdown(
        f"<div class='score-box {color_class}'>"
        f"<h2 style='color:#3333;'>Batch Average Score: {avg_score:.2f}</h2>"
        f"<p style='font-size:18px; color:#3333;'>Quality Level: {quality_level}</p>"
        f"<p style='font-size:16px; color:#3333;'>Total Chats: {len(results)}</p>"
        f"</div>",
        unsafe_allow_html=True
    )
    
    # Create a simplified summary table
    st.subheader("Summary of All Chat Scores")
    
    # Prepare data for table - simplified
    summary_data = []
    for i, result in enumerate(results):
        chat_id = result.get('chat_id', f"Chat_{i+1}")
        score = result.get('weighted_overall_score', 0)
        language = result.get('detected_language', 'Unknown')
        
        # Determine quality level for this chat
        chat_quality = "Unknown"
        if "scoring_system" in rules and "quality_levels" in rules["scoring_system"]:
            for level in rules["scoring_system"]["quality_levels"]:
                if level["range"]["min"] <= score <= level["range"]["max"]:
                    chat_quality = level["name"]
                    break
        
        summary_data.append({
            "Chat": chat_id,
            "Overall Score": f"{score:.2f}",
            "Quality": chat_quality,
            "Language": language
        })
    
    # Display table
    st.table(summary_data)
    
    # Create Detailed CSV with explanations, examples, and suggestions
    detailed_csv_data = "Chat ID,Parameter,Score,Explanation,Example,Suggestion\n"
    
    for i, result in enumerate(results):
        chat_id = result.get('chat_id', f"Chat_{i}")
        
        # Add data for each parameter
        for param in rules["parameters"]:
            param_name = param["name"]
            if param_name in result:
                param_data = result[param_name]
                score = param_data.get("score", "N/A")
                
                # Clean and escape text fields for CSV
                explanation = str(param_data.get("explanation", "")).replace('"', '""')
                example = str(param_data.get("example", "")).replace('"', '""')
                suggestion = str(param_data.get("suggestion", "N/A")).replace('"', '""')
                
                # Handle None values
                if suggestion == "None" or suggestion == "null":
                    suggestion = "N/A"
                
                detailed_csv_data += f'"{chat_id}","{param_name}",{score},"{explanation}","{example}","{suggestion}"\n'
    
    # Single download option
    st.subheader("Download Report")
    
    # Create file-like object for the detailed CSV
    detailed_csv_file = io.BytesIO(detailed_csv_data.encode('utf-8'))
    
    # Centered single download button
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.download_button(
            label="Download Detailed Analysis (CSV)",
            data=detailed_csv_file,
            file_name=f"qa_detailed_{timestamp}_{batch_id}.csv",
            mime="text/csv",
            key=f"detailed_dl_{batch_id}"
        )
