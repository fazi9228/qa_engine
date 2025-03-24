import streamlit as st
import json
from datetime import datetime
import concurrent.futures

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

# Function to analyze a transcript with cultural considerations
def analyze_chat_transcript(transcript, rules, target_language="en", prompt_template_path="QA_prompt.md", model_provider="anthropic", model_name=None):
    """
    Analyze a text chat transcript using the specified model and prompt template.
    
    Args:
        transcript (str): The chat transcript text
        rules (dict): Evaluation rules dictionary
        target_language (str): Target language for analysis response ('en' or 'native')
        prompt_template_path (str): Path to the prompt template file
        model_provider (str): AI provider ("anthropic" or "openai")
        model_name (str): Model name to use
        
    Returns:
        dict: Analysis results or None if analysis failed
    """
    try:
        # Set default model name if not provided
        if not model_name:
            if model_provider == "anthropic":
                model_name = "claude-3-7-sonnet-20250219"
            elif model_provider == "openai":
                model_name = "gpt-4o"
        
        # Use cached language detection for better performance
        text_sample = transcript[:200]  # Use a sample to reduce cache size
        lang_code, lang_name = detect_language_cached(text_sample, model_provider)
        
        # Extract parameter names and descriptions from rules
        parameters_list = ""
        for param in rules["parameters"]:
            parameters_list += f"- {param['name']}: {param['description']}\n"
        
        # Extract scoring scale information
        scale_max = 100
        if "scoring_system" in rules and "score_scale" in rules["scoring_system"]:
            scale_max = rules["scoring_system"]["score_scale"]["max"]
        
        # Get scoring system information for the prompt
        scoring_info = ""
        if "scoring_system" in rules and "quality_levels" in rules["scoring_system"]:
            scoring_info = "Use the following scoring scale:\n"
            for level in rules["scoring_system"]["quality_levels"]:
                scoring_info += f"- {level['name']} ({level['range']['min']}-{level['range']['max']}): {level['description']}\n"
        
        # Load prompt template from file
        prompt_template = load_prompt_template(prompt_template_path)
        
        response_text = ""
        
        if model_provider == "anthropic":
            # Initialize client when needed
            client = initialize_anthropic_client()
            if not client:
                st.error("Anthropic API key is required for Claude analysis.")
                return None
                
            # Create a simplified system prompt for Claude that includes exact parameter names
            system_prompt = f"""You are a customer support QA analyst for Pepperstone, a forex broker.

You must score the transcript for EXACTLY these parameters on a scale of 0-{scale_max}:
{parameters_list}

IMPORTANT: The transcript may contain anonymized sensitive information in the format [TYPE]. 
For example, [AMOUNT], [EMAIL], [PHONE], [TRANSACTION_ID], etc. 
These represent real sensitive data that has been removed for privacy. 
When evaluating the transcript, consider these tags as if they were the actual information.

{prompt_template}

## Scoring System
{scoring_info}

## Response Format
Your evaluation MUST be formatted as a valid JSON object with the following structure:
```json
{{
  "Parameter Name": {{
    "score": 85,
    "explanation": "Brief explanation here",
    "example": "Example from transcript here",
    "suggestion": null
  }},
  ... for each parameter
}}
```
The 'suggestion' field should be null if score is 80 or higher.
YOUR ENTIRE RESPONSE MUST BE VALID JSON WITH NO ADDITIONAL TEXT.
"""

            user_prompt = f"""Analyze this customer support transcript and score EACH parameter listed in my instructions:

{transcript}

Remember to format your response as a valid JSON object with EXACT parameter names as keys, and each value being 
an object with "score", "explanation", "example", and "suggestion" fields.
The suggestion field should be null if score is 80 or higher.

Return your explanation, examples, and suggestions in {lang_name if target_language == 'native' else 'English'}.
"""
            
            try:
                with st.sidebar.expander("Debug - Prompt", expanded=False):
                    st.text("System prompt (excerpt):")
                    st.text(system_prompt[:500] + "...")
                
                response = client.messages.create(
                    model=model_name,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.0
                )
                response_text = response.content[0].text
                
                with st.sidebar.expander("Debug - Response", expanded=False):
                    st.text("Raw API response (excerpt):")
                    st.text(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                
            except Exception as claude_error:
                st.error(f"Claude API error: {str(claude_error)}")
                return None
            
        elif model_provider == "openai":
            # Initialize client when needed
            client = initialize_openai_client()
            if not client:
                st.error("OpenAI API key is required for GPT analysis.")
                return None
                
            # Create a simplified system prompt for OpenAI that includes exact parameter names
            system_prompt = f"""You are a QA analyst for Pepperstone, a forex broker. Score the support transcript."""
            
            # More structured user prompt with exact parameter names
            user_prompt = f"""Score this support transcript on a scale of 0-{scale_max} for EXACTLY these parameters:
{parameters_list}

NOTE: This transcript may contain anonymized data represented as [TYPE] tags (like [AMOUNT], [PHONE], etc.).
Treat these as if they were the actual information when evaluating the conversation.

TRANSCRIPT:
{transcript}

Format your response as a JSON object exactly like this:
{{
  "Parameter Name": {{
    "score": 85,
    "explanation": "Brief explanation",
    "example": "Example from transcript",
    "suggestion": null
  }},
  ... for each parameter
}}

Include suggestion only when score is below 80, otherwise set to null.
"""
            try:
                with st.sidebar.expander("Debug - Prompt", expanded=False):
                    st.text("User prompt (excerpt):")
                    st.text(user_prompt[:500] + "...")
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                
                response_text = response.choices[0].message.content
                
                with st.sidebar.expander("Debug - Response", expanded=False):
                    st.text("Raw API response (excerpt):")
                    st.text(response_text[:500] + "..." if len(response_text) > 500 else response_text)
                
            except Exception as openai_error:
                st.error(f"OpenAI API error: {str(openai_error)}")
                return None
        else:
            st.error(f"Unsupported model provider: {model_provider}")
            return None
        
        # Parse the response with improved error handling
        analysis = parse_json_response(response_text)
        
        if not analysis:
            st.error("Failed to parse API response to JSON")
            return None
        
        # Check if we need to scale scores (if API returns 0-10 but we expect 0-100)
        if "scoring_system" in rules and "score_scale" in rules["scoring_system"]:
            expected_max = 100
            actual_max = rules["scoring_system"]["score_scale"]["max"]
            
            if actual_max == 10 and expected_max == 100:
                # We need to scale up scores from 0-10 to 0-100
                with st.sidebar.expander("Debug - Score Scaling", expanded=False):
                    st.text(f"Scaling scores from 0-{actual_max} to 0-{expected_max}")
                
                # Scale all parameter scores
                for param_name in analysis:
                    if param_name != "weighted_overall_score" and isinstance(analysis[param_name], dict) and "score" in analysis[param_name]:
                        orig_score = analysis[param_name]["score"]
                        if isinstance(orig_score, (int, float)) and orig_score <= actual_max:
                            analysis[param_name]["score"] = orig_score * 10
        
        # Calculate weighted score
        total_weight = sum(param["weight"] for param in rules["parameters"])
        weighted_score = 0
        
        # Track missing parameters for debugging
        missing_params = []
        
        for param in rules["parameters"]:
            param_name = param["name"]
            if param_name in analysis:
                weighted_score += analysis[param_name]["score"] * param["weight"]
            else:
                missing_params.append(param_name)
        
        if missing_params:
            with st.sidebar.expander("Debug - Missing Parameters", expanded=False):
                st.warning(f"Parameters missing from API response: {', '.join(missing_params)}")
        
        weighted_score = weighted_score / total_weight if total_weight > 0 else 0
        analysis["weighted_overall_score"] = round(weighted_score, 2)
        
        # Add model provider info to the result
        analysis["model_provider"] = model_provider
        analysis["model_name"] = model_name
        
        with st.sidebar.expander("Debug - Final Analysis", expanded=False):
            st.json(analysis)
        
        return analysis
    
    except Exception as e:
        st.error(f"Error analyzing transcript: {str(e)}")
        st.exception(e)  # Show the full exception for debugging
        return None
    
# Function to visualize chat analysis results
def visualize_chat_results(result, rules):
    """
    Visualize the chat analysis results
    
    Args:
        result (dict): Analysis result dictionary
        rules (dict): Evaluation rules dictionary
    """
    if not result:
        return
    
    # Display the overall score with appropriate color based on quality level
    overall_score = result.get('weighted_overall_score', 0)
    
    # Determine quality level based on score ranges
    quality_level = "Unknown"
    
    # Use scoring system from rules if available
    if "scoring_system" in rules and "quality_levels" in rules["scoring_system"]:
        for level in rules["scoring_system"]["quality_levels"]:
            if level["range"]["min"] <= overall_score <= level["range"]["max"]:
                quality_level = level["name"]
                break
    # Fallback to score_ranges if available
    elif "score_ranges" in rules:
        for level_name, level_range in rules["score_ranges"].items():
            if level_range["min"] <= overall_score <= level_range["max"]:
                quality_level = level_name.capitalize()
                break
    # Use simple quality level determination as final fallback
    else:
        if overall_score >= 90:
            quality_level = "Excellent"
        elif overall_score >= 80:
            quality_level = "Good"
        elif overall_score >= 70:
            quality_level = "Needs Improvement"
        else:
            quality_level = "Poor"
    
    # Set color class based on quality level
    color_class = "score-box-needs-improvement"  # Default
    if "excellent" in quality_level.lower():
        color_class = "score-box-excellent"
    elif "good" in quality_level.lower():
        color_class = "score-box-good"
    elif "poor" in quality_level.lower():
        color_class = "score-box-poor"
    
    # Add smiley based on score
    smiley = "😀" if overall_score >= 85 else "🙂" if overall_score >= 70 else "😐" if overall_score >= 50 else "😟"
    
    st.markdown(
    f"<div class='score-box {color_class}'>"
    f"<h2 style='color:#333333;'>Overall Score: {overall_score:.2f} {smiley}</h2>"
    f"<p style='font-size:18px; color:#333333;'>Quality Level: {quality_level}</p>"
    f"</div>",
    unsafe_allow_html=True
    )
    
    # Display language information
    st.info(f"Transcript Language: {result.get('detected_language', 'Unknown')}")
    
    # Add download button for results
    # Create a formatted JSON string for download
    json_result = json.dumps(result, indent=2, ensure_ascii=False)
    
    # Create a CSV of parameter scores for download
    csv_data = "Parameter,Score,Has Suggestion\n"
    for param in rules["parameters"]:
        param_name = param["name"]
        if param_name in result and isinstance(result[param_name], dict):
            score = result[param_name].get("score", "N/A")
            has_suggestion = "Yes" if result[param_name].get("suggestion") else "No"
            csv_data += f'"{param_name}",{score},{has_suggestion}\n'

    # Add download options in columns
    col1, col2 = st.columns(2)
    with col1:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Convert to bytes with UTF-8 encoding to preserve characters
        json_bytes = json_result.encode('utf-8')
        
        st.download_button(
            label="Download Analysis (JSON)",
            data=json_bytes,
            file_name=f"qa_analysis_{timestamp}.json",
            mime="application/json",
            key=f"download_json_{timestamp}"
        )

    # The CSV part remains mostly the same, but let's ensure UTF-8 there too
    with col2:
        # Ensure CSV is also properly encoded
        csv_bytes = csv_data.encode('utf-8')
        
        st.download_button(
            label="Download Scores (CSV)",
            data=csv_bytes,
            file_name=f"qa_scores_{timestamp}.csv",
            mime="text/csv",
            key=f"download_csv_{timestamp}"
        )
    
    # Check if we have categories
    has_categories = "categories" in rules
    
    if has_categories:
        # Create a tab for each category
        category_tabs = st.tabs([cat["name"] for cat in rules["categories"]])
        
        for i, category in enumerate(rules["categories"]):
            with category_tabs[i]:
                st.subheader(f"{category['name']} Parameters")
                
                # Display scores for each parameter in this category
                for param_name in category["parameters"]:
                    if param_name in result:
                        score = result[param_name]["score"]
                        
                        # Use fixed color without smiley indicator
                        st.markdown(f"<h3>{param_name}: {score}</h3>", unsafe_allow_html=True)
                        st.markdown(f"**Explanation**: {result[param_name]['explanation']}")
                        
                        # Only show example if it exists and isn't empty or "N/A"
                        example = result[param_name].get('example', '')
                        if example and example.lower() != 'n/a':
                            st.markdown(f"**Example**: _{example}_")
                        
                        # Only show suggestion if it exists and isn't null or empty
                        suggestion = result[param_name].get('suggestion')
                        if suggestion:
                            st.markdown(f"**Suggestion**: {suggestion}")
                        
                        st.markdown("---")
                    else:
                        # If parameter is missing in results, show placeholder
                        st.warning(f"No score available for: {param_name}")
                        st.markdown("---")
    else:
        # Display individual parameter scores without categories
        st.subheader("Parameter Scores")
        for param in rules["parameters"]:
            param_name = param["name"]
            if param_name in result:
                score = result[param_name]["score"]
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # No smiley here
                    st.markdown(f"### {score}")
                
                with col2:
                    st.markdown(f"### {param_name}")
                    st.markdown(f"**Explanation**: {result[param_name]['explanation']}")
                    
                    # Only show example if it exists and isn't empty or "N/A"
                    example = result[param_name].get('example', '')
                    if example and example.lower() != 'n/a':
                        st.markdown(f"**Example**: _{example}_")
                    
                    # Only show suggestion if it exists and isn't null or empty
                    suggestion = result[param_name].get('suggestion')
                    if suggestion:
                        st.markdown(f"**Suggestion**: {suggestion}")
                
                st.markdown("---")
            else:
                # If parameter is missing in results, show placeholder
                st.warning(f"No score available for: {param_name}")
                st.markdown("---")

# Function to create the chat transcript analysis UI
def render_chat_analysis_ui():
    """
    Renders the chat transcript analysis UI components
    """
    st.header("Chat Transcript Analysis")
    
    # Text area for transcript input
    single_transcript = st.text_area(
        "Paste a customer support chat transcript",
        height=300,
        placeholder="Customer: Hi, I'm having trouble with my account...\nAgent: Hello! I'd be happy to help you with your account issue..."
    )
    
    prompt_path = "QA_prompt.md"  # Required file, no fallback
    
    # Return the transcript text for processing
    return single_transcript, prompt_path

# Function to process chat analysis
def process_chat_analysis(transcript, rules, target_language="en", prompt_path="QA_prompt.md", 
                         provider_internal_name="anthropic", model_name=None, debug_mode=False):
    """
    Process a chat transcript analysis request
    
    Args:
        transcript (str): The chat transcript text
        rules (dict): Evaluation rules dictionary
        target_language (str): Target language for analysis ('en' or 'native')
        prompt_path (str): Path to the prompt template
        provider_internal_name (str): Provider name ("anthropic" or "openai")
        model_name (str): Model name
        debug_mode (bool): Whether debug mode is enabled
        
    Returns:
        dict: Analysis result or None if analysis failed
    """
    if not transcript:
        st.warning("Please paste a transcript to analyze")
        return None
        
    # Create a single progress container instead of multiple placeholders
    progress_container = st.container()
    result_container = st.container()
    
    # Simple progress bar approach
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Stage 1: Processing transcript
    status_text.text("Processing transcript...")
    progress_bar.progress(10)
    
    # Add a debug output for the transcript
    if debug_mode:
        with st.sidebar:
            with st.expander("Debug - Transcript", expanded=False):
                st.text(transcript[:200] + "..." if len(transcript) > 200 else transcript)
    
    # Stage 2: Language detection
    status_text.text("Detecting language...")
    progress_bar.progress(30)
    
    # Determine language settings
    try:
        # Get language detection from main UI setting
        auto_detect = st.session_state.get("auto_detect", True)
        
        if auto_detect:
            # Use cached language detection
            text_sample = transcript[:200]
            lang_code, lang_name = detect_language_cached(text_sample, provider_internal_name)
            # Add detected language to sidebar status
            with st.sidebar:
                with st.expander("Status Messages", expanded=False):
                    st.info(f"Detected language: {lang_name}")
        else:
            lang_name = st.session_state.get("transcript_language", "English")
            
        progress_bar.progress(50)
        
        # Stage 3: Analysis
        model_provider_display = "Anthropic Claude" if provider_internal_name == "anthropic" else "OpenAI GPT"
        status_text.text(f"Analyzing with {model_provider_display}...")
        
        # Analyze the transcript using the prompt template
        result = analyze_chat_transcript(
            transcript, 
            rules, 
            target_language,
            prompt_template_path=prompt_path,
            model_provider=provider_internal_name,
            model_name=model_name
        )
        
        progress_bar.progress(90)
        
        # Store transcript and result in session state to prevent reset on download
        st.session_state.current_transcript = transcript
        st.session_state.current_result = result
        
        # Add language information to result
        if result:
            result["detected_language"] = lang_name
            
            # Update progress
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Display visualization
            with result_container:
                try:
                    visualize_chat_results(result, rules)
                    st.caption(f"Analysis performed by {result.get('model_provider', provider_internal_name).title()} {result.get('model_name', model_name)}")
                except Exception as viz_error:
                    with st.sidebar:
                        with st.expander("Status Messages", expanded=False):
                            st.error(f"Error visualizing results: {str(viz_error)}")
                    st.json(result)  # Just show the raw result
            
            return result
        else:
            progress_bar.progress(100)
            status_text.text("Analysis failed.")
            with result_container:
                st.error("Analysis failed. Please check the error messages in the Status Messages section.")
            return None
                
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("Error during processing.")
        with st.sidebar:
            with st.expander("Status Messages", expanded=False):
                st.error(f"Error during processing: {str(e)}")
                if debug_mode:
                    st.exception(e)
        with result_container:
            st.error("Analysis failed. Please check the error messages in the Status Messages section.")
        return None