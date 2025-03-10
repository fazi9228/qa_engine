import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import anthropic
from datetime import datetime
import time

# Add OpenAI import
try:
    import openai
except ImportError:
    pass

# Silence stderr to prevent "No secrets found" messages
import sys
import io

# Temporarily redirect stderr during import
old_stderr = sys.stderr
sys.stderr = io.StringIO()

# Load environment variables from .env file in local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Restore stderr
sys.stderr = old_stderr

# Page configuration
st.set_page_config(
    page_title="Customer Support QA Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .score-excellent { color: green; }
    .score-good { color: blue; }
    .score-needs-improvement { color: orange; }
    .score-poor { color: red; }
    
    .score-box {
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced API key handling with multiple fallback mechanisms
def get_api_key(provider="anthropic"):
    """
    Get the API key from various sources in the following priority order:
    1. Environment variable
    2. Streamlit secrets (for deployment)
    3. Session state (if manually entered in UI)
    
    Args:
        provider (str): The API provider ("anthropic" or "openai")
    """
    env_var_names = {
        "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "openai": ["OPENAI_API_KEY"]
    }
    
    session_state_key = {
        "anthropic": "user_anthropic_api_key",
        "openai": "user_openai_api_key"
    }
    
    # Check environment variables first (loaded from .env file in local development)
    for env_var in env_var_names.get(provider, []):
        api_key = os.getenv(env_var)
        if api_key:
            return api_key
    
    # Check Streamlit secrets (for cloud deployment)
    try:
        if provider in st.secrets:
            api_key = st.secrets[provider]["api_key"]
            if api_key:
                return api_key
    except Exception:
        # Silently continue if secrets aren't available
        pass
    
    # Check session state (if user manually entered it)
    state_key = session_state_key.get(provider)
    if state_key in st.session_state and st.session_state[state_key]:
        return st.session_state[state_key]
    
    # No API key found
    return None

# Initialize Anthropic client
def initialize_anthropic_client():
    api_key = get_api_key("anthropic")
    if api_key:
        try:
            return anthropic.Anthropic(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing Anthropic client: {str(e)}")
            return None
    return None

# Initialize OpenAI client
def initialize_openai_client():
    api_key = get_api_key("openai")
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            # Test the client with a simple request
            test_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=5
            )
            # If we get here, the client is working
            return client
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    return None

def check_password():
    """
    Returns `True` if the user had the correct password.
    """
    # Load password from .env file or environment variables
    correct_password = os.getenv("QA_PASSWORD")
    
    if not correct_password:
        st.error("No password set in environment variables. Set QA_PASSWORD in .env file.")
        return False
    
    # Create a session state object if it doesn't exist yet
    if "password_entered" not in st.session_state:
        st.session_state.password_entered = False
    
    if "password_attempts" not in st.session_state:
        st.session_state.password_attempts = 0

    if st.session_state.password_entered:
        return True

    # Show input for password
    st.markdown("# Customer Support QA Engine Login")
    password = st.text_input("Enter the application password", type="password")
    
    if password:
        if password == correct_password:
            st.session_state.password_entered = True
            st.rerun()  # Changed from st.experimental_rerun()
            return True
        else:
            st.session_state.password_attempts += 1
            if st.session_state.password_attempts >= 3:
                st.error("Too many incorrect password attempts. Please try again later.")
                time.sleep(3)  # Add a delay to discourage brute force attempts
            else:
                st.error(f"Incorrect password. Attempts: {st.session_state.password_attempts}/3")
            return False
    else:
        return False

# Helper function to parse JSON from API responses with better error handling
def parse_json_response(response_text):
    """Parse JSON from API responses with better error handling"""
    try:
        # Try direct JSON parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try one more approach - look for ```json blocks
                import re
                json_blocks = re.findall(r'```json\s*([\s\S]*?)\s*```', response_text)
                if json_blocks:
                    try:
                        return json.loads(json_blocks[0])
                    except json.JSONDecodeError:
                        st.error("Failed to extract valid JSON from API response")
                        st.code(response_text, language="text")
                        st.write("Raw response from API:")
                        st.write(response_text)
                        return None
                else:
                    st.error("Failed to extract valid JSON from API response")
                    st.code(response_text, language="text")
                    st.write("Raw response from API:")
                    st.write(response_text)
                    return None
        else:
            st.error("Could not find JSON structure in API response")
            st.write("Raw response from API:")
            st.write(response_text)
            return None
        
# Function to detect language
def detect_language(text, model_provider="anthropic"):
    try:
        # Use the model provider chosen in the sidebar if available
        if "model_provider" in st.session_state:
            if st.session_state.model_provider == "OpenAI GPT":
                model_provider = "openai"
            else:
                model_provider = "anthropic"
                
        # Check if API key is available for selected provider
        if model_provider == "anthropic":
            client = initialize_anthropic_client()
            if not client:
                st.warning("Anthropic API key is required for language detection. Falling back to OpenAI if available.")
                model_provider = "openai"
                client = initialize_openai_client()
                if not client:
                    st.warning("No API keys available for language detection. Using English as default.")
                    return "en", "English (default)"
        else:  # openai
            client = initialize_openai_client()
            if not client:
                st.warning("OpenAI API key is required for language detection. Falling back to Anthropic if available.")
                model_provider = "anthropic"
                client = initialize_anthropic_client()
                if not client:
                    st.warning("No API keys available for language detection. Using English as default.")
                    return "en", "English (default)"
        
        # Use selected provider to detect language
        if model_provider == "anthropic":
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=10,
                system="You are a language detection tool. Respond with only the ISO language code.",
                messages=[
                    {"role": "user", "content": f"Detect the language of this text and respond with only the ISO language code (e.g., 'en', 'zh-CN', 'zh-TW', 'vi', 'th', 'pt-ES', 'pt-BR'): {text[:200]}..."}
                ]
            )
            lang_code = response.content[0].text.strip().lower()
        
        elif model_provider == "openai":
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=10,
                messages=[
                    {"role": "system", "content": "You are a language detection tool. Respond with only the ISO language code."},
                    {"role": "user", "content": f"Detect the language of this text and respond with only the ISO language code (e.g., 'en', 'zh-CN', 'zh-TW', 'vi', 'th', 'pt-ES', 'pt-BR'): {text[:200]}..."}
                ]
            )
            lang_code = response.choices[0].message.content.strip().lower()
        
        # Map language codes to full names
        lang_map = {
            "en": "English",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "vi": "Vietnamese",
            "th": "Thai",
            "pt-es": "Portuguese (Spain)",
            "pt-br": "Portuguese (Brazil)",
            "es": "Spanish"
        }
        
        return lang_code, lang_map.get(lang_code, "Unknown")
    
    except Exception as e:
        st.warning(f"Could not detect language automatically: {str(e)}")
        return "en", "English (default)"

# Function to load the prompt template from a markdown file
def load_prompt_template(file_path="QA_prompt.md"):
    """
    Load the QA prompt template from a markdown file
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template
    except FileNotFoundError:
        st.error(f"Prompt template file '{file_path}' not found. This file is required.")
        st.stop()

# Function to load evaluation rules from JSON file
@st.cache_data
def load_evaluation_rules(file_path="evaluation_rules.json", scoring_path="scoring_system.json"):
    try:
        # Load main evaluation rules
        with open(file_path, "r") as f:
            rules = json.load(f)
        
        # Load scoring system if available
        try:
            with open(scoring_path, "r") as f:
                scoring = json.load(f)
                # Merge scoring system into rules
                rules["scoring_system"] = scoring.get("scoring_system", {})
                
                # Check and standardize scoring scale
                if "score_scale" in rules["scoring_system"]:
                    scale = rules["scoring_system"]["score_scale"]
                    
                    # If scale is 0-10, normalize quality levels to match
                    if scale["max"] == 10:
                        with st.sidebar.expander("Debug - Score Scale", expanded=False):
                            st.info(f"Detected 0-10 scale in scoring system")
                            
                        # No need to modify - we'll handle the conversion during analysis
                        pass
                
        except FileNotFoundError:
            # Require scoring system file
            st.error(f"Scoring system file '{scoring_path}' not found.")
            st.stop()
            
        return rules
    except FileNotFoundError:
        # Require evaluation rules file
        st.error(f"Evaluation rules file '{file_path}' not found.")
        st.stop()

# Function to analyze a transcript with cultural considerations
def analyze_transcript(transcript, rules, target_language="en", prompt_template_path="QA_prompt.md", model_provider="anthropic", model_name=None):
    try:
        # Initialize client based on provider
        client = None
        if model_provider == "anthropic":
            client = initialize_anthropic_client()
            if not client:
                st.error("Anthropic API key is required for Claude analysis.")
                return None
        elif model_provider == "openai":
            client = initialize_openai_client()
            if not client:
                st.error("OpenAI API key is required for GPT analysis.")
                return None
        else:
            st.error(f"Unsupported model provider: {model_provider}")
            return None
            
        # Set default model name if not provided
        if not model_name:
            if model_provider == "anthropic":
                model_name = "claude-3-7-sonnet-20250219"
            elif model_provider == "openai":
                model_name = "gpt-4o"
        
        # Detect language of the transcript
        lang_code, lang_name = detect_language(transcript, model_provider)
        
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
            # Create a simplified system prompt for Claude that includes exact parameter names
            system_prompt = f"""You are a customer support QA analyst for Pepperstone, a forex broker.

You must score the transcript for EXACTLY these parameters on a scale of 0-{scale_max}:
{parameters_list}

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
            # Create a simplified system prompt for OpenAI that includes exact parameter names
            system_prompt = f"""You are a QA analyst for Pepperstone, a forex broker. Score the support transcript."""
            
            # More structured user prompt with exact parameter names
            user_prompt = f"""Score this support transcript on a scale of 0-{scale_max} for EXACTLY these parameters:
{parameters_list}

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
    
def convert_score(score, from_scale, to_scale):
    """
    Convert a score from one scale to another
    
    Args:
        score (float): The score to convert
        from_scale (tuple): The source scale as (min, max)
        to_scale (tuple): The target scale as (min, max)
    
    Returns:
        float: The converted score
    """
    from_min, from_max = from_scale
    to_min, to_max = to_scale
    
    # Ensure score is within source scale bounds
    score = max(from_min, min(score, from_max))
    
    # Convert to percentage within source scale
    percentage = (score - from_min) / (from_max - from_min)
    
    # Apply percentage to target scale
    converted = to_min + percentage * (to_max - to_min)
    
    return converted

# Function to visualize results
# Function to visualize results
def visualize_results(result, rules):
    if not result:
        return
    
    # Display the overall score with a light gray color
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
    
    # Use a light gray color
    color = "#e0e0e0"  # Light gray color
    
    # Add smiley based on score (only for overall score)
    smiley = "😀" if overall_score >= 85 else "🙂" if overall_score >= 70 else "😐" if overall_score >= 50 else "😟"
    
    st.markdown(
        f"<div class='score-box' style='background-color:{color};'>"
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
        csv_data = "Parameter,Score,Has Suggestion\n"
        for param in rules["parameters"]:
            param_name = param["name"]
            if param_name in result and isinstance(result[param_name], dict):
                score = result[param_name].get("score", "N/A")
                has_suggestion = "Yes" if result[param_name].get("suggestion") else "No"
                csv_data += f'"{param_name}",{score},{has_suggestion}\n'
        
        # Convert to bytes with UTF-8 encoding
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
                
# Main application
# Main application
def main():
    # First check password before showing any app content
    if not check_password():
        return
    
    # Update the title with an icon
    st.title("Customer Support QA Engine Beta 🔍")
    
    # Check for required files before proceeding
    required_files = {
        "evaluation_rules.json": "Evaluation Rules",
        "scoring_system.json": "Scoring System",
        "QA_prompt.md": "QA Prompt Template"
    }
    
    missing_files = []
    for file_path, file_desc in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{file_desc} ({file_path})")
    
    if missing_files:
        st.error("Required files not found:")
        for missing_file in missing_files:
            st.error(f"- {missing_file}")
        st.info("Please ensure all required files are available in the application directory.")
        st.stop()
    
    # Create a container in the sidebar for status messages
    with st.sidebar:
        status_container = st.empty()
    
    # Debug mode toggle (hidden in sidebar expander for convenience)
    with st.sidebar.expander("Developer Options", expanded=False):
        debug_mode = st.checkbox("Debug Mode", value=False)
    
    try:
        # Load evaluation rules
        rules = load_evaluation_rules()
        # Display success message in sidebar instead of main
        with st.sidebar:
            status_expander = st.expander("Status Messages", expanded=False)
            with status_expander:
                st.success("Loaded evaluation rules successfully")

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
            provider_internal_name = "anthropic" if model_provider == "Anthropic Claude" else "openai"
            
            # API Key handling (dynamically based on selected provider)
            if provider_internal_name == "anthropic":
                model_name = "claude-3-7-sonnet-20250219"
                st.info("Using Claude 3.7 Sonnet")
                
                # Show API key input for Anthropic
                st.subheader("Anthropic API Key")
                anthropic_api_key = get_api_key("anthropic")
                
                if not anthropic_api_key:
                    user_api_key = st.text_input("Enter your Anthropic API key:", type="password")
                    if user_api_key:
                        st.session_state.user_anthropic_api_key = user_api_key
                        st.rerun()
                else:
                    st.success("✅ Anthropic API key available")
            else:  # OpenAI
                model_name = "gpt-4o"
                st.info("Using GPT-4o")
                
                # Show API key input for OpenAI
                st.subheader("OpenAI API Key")
                openai_api_key = get_api_key("openai")
                
                if not openai_api_key:
                    user_api_key = st.text_input("Enter your OpenAI API key:", type="password")
                    if user_api_key:
                        st.session_state.user_openai_api_key = user_api_key
                        st.rerun()
                else:
                    st.success("✅ OpenAI API key available")
            
            # Language settings
            st.subheader("Language Settings")
            auto_detect = st.checkbox("Auto-detect transcript language", value=True)
            
            if not auto_detect:
                transcript_language = st.selectbox(
                    "Transcript language",
                    ["English", "Chinese (Simplified)", "Chinese (Traditional)", "Vietnamese", "Thai", 
                    "Portuguese (Spain)", "Portuguese (Brazil)", "Spanish"]
                )
                    
            output_language = st.radio(
                "Output language for analysis",
                ["English", "Native language of transcript"]
            )
            
            # Convert output_language to parameter for function
            target_language = "native" if output_language == "Native language of transcript" else "en"
            
            # About section
            st.markdown("---")
            st.caption("Customer Support QA Engine Beta")
            st.caption("Powered by Anthropic Claude and OpenAI")

        # Test API connections
        anthropic_client = None
        openai_client = None
        
        # Test Anthropic connection
        anthropic_api_key = get_api_key("anthropic")
        if anthropic_api_key:
            try:
                anthropic_client = initialize_anthropic_client()
                if anthropic_client:
                    # Display success in sidebar instead of main
                    with st.sidebar:
                        with status_expander:
                            st.success("Successfully connected to Anthropic API")
            except Exception as e:
                with st.sidebar:
                    with status_expander:
                        st.error(f"Error connecting to Anthropic API: {str(e)}")
        
        # Test OpenAI connection
        openai_api_key = get_api_key("openai")
        if openai_api_key:
            try:
                openai_client = initialize_openai_client()
                if openai_client:
                    # Display success in sidebar instead of main
                    with st.sidebar:
                        with status_expander:
                            st.success("Successfully connected to OpenAI API")
            except Exception as e:
                with st.sidebar:
                    with status_expander:
                        st.error(f"Error connecting to OpenAI API: {str(e)}")
        
        # Check if any API key is available
        if not anthropic_client and not openai_client:
            st.warning("⚠️ Please provide at least one valid API key to use this application.")
            st.info("Enter your API key in the sidebar.")
            return
        
        # If the selected provider's API key is not available, but the other one is,
        # automatically switch to the available provider
        if provider_internal_name == "anthropic" and not anthropic_client and openai_client:
            with st.sidebar:
                st.warning("Anthropic API key not available, switching to OpenAI")
            provider_internal_name = "openai"
            model_name = "gpt-4o"
            model_provider = "OpenAI GPT"
        elif provider_internal_name == "openai" and not openai_client and anthropic_client:
            with st.sidebar:
                st.warning("OpenAI API key not available, switching to Anthropic")
            provider_internal_name = "anthropic"
            model_name = "claude-3-7-sonnet-20250219"
            model_provider = "Anthropic Claude"

        # Main content area - only display if we have a valid API key
        st.markdown("Upload chat transcripts for AI-powered quality assessment")

        # Add model info
        st.info(f"Using {model_provider} model: {model_name}")

        # Single Analysis
        st.header("Transcript Analysis")
        single_transcript = st.text_area(
            "Paste a customer support chat transcript",
            height=300,
            placeholder="Customer: Hi, I'm having trouble with my account...\nAgent: Hello! I'd be happy to help you with your account issue..."
        )

        prompt_path = "QA_prompt.md"  # Required file, no fallback

        if st.button("Analyze Transcript", key="analyze_single"):
            if not single_transcript:
                st.warning("Please paste a transcript to analyze")
            else:
                with st.spinner(f"Analyzing transcript with {model_provider}..."):
                    # Add a debug output for the transcript - in sidebar
                    if debug_mode:
                        with st.sidebar:
                            with st.expander("Debug - Transcript", expanded=False):
                                st.text(single_transcript[:200] + "..." if len(single_transcript) > 200 else single_transcript)
                        
                    # Determine language settings
                    if auto_detect:
                        try:
                            lang_code, lang_name = detect_language(single_transcript, provider_internal_name)
                            # Add detected language to sidebar status instead
                            with st.sidebar:
                                with status_expander:
                                    st.info(f"Detected language: {lang_name}")
                        except Exception as e:
                            with st.sidebar:
                                with status_expander:
                                    st.error(f"Error detecting language: {str(e)}")
                            lang_name = "English (default after error)"
                    else:
                        lang_name = transcript_language
                        
                    # Analyze the transcript using the prompt template
                    try:
                        result = analyze_transcript(
                            single_transcript, 
                            rules, 
                            target_language,
                            prompt_template_path=prompt_path,
                            model_provider=provider_internal_name,
                            model_name=model_name
                        )
                        
                        # Store transcript and result in session state to prevent reset on download
                        st.session_state.current_transcript = single_transcript
                        st.session_state.current_result = result
                        
                    except Exception as analysis_error:
                        with st.sidebar:
                            with status_expander:
                                st.error(f"Error during analysis: {str(analysis_error)}")
                                if debug_mode:
                                    st.exception(analysis_error)
                        result = None
                    
                    # Add language information to result
                    if result:
                        result["detected_language"] = lang_name
                        # Display visualization
                        try:
                            visualize_results(result, rules)
                        except Exception as viz_error:
                            with st.sidebar:
                                with status_expander:
                                    st.error(f"Error visualizing results: {str(viz_error)}")
                            st.json(result)  # Just show the raw result
                        
                        # Show model info in result
                        st.caption(f"Analysis performed by {result.get('model_provider', provider_internal_name).title()} {result.get('model_name', model_name)}")
                    else:
                        st.error("Analysis failed. Please check the error messages in the Status Messages section.")
        
        # Check if we have a stored result from previous analysis
        elif "current_result" in st.session_state and "current_transcript" in st.session_state:
            # Use the stored result
            result = st.session_state.current_result
            
            # Optional: Display a note that we're showing previous results
            st.info("Showing analysis for previously submitted transcript")
            
            if result:
                # Visualize using the stored result
                visualize_results(result, rules)
                
                # Show model info in result
                st.caption(f"Analysis performed by {result.get('model_provider', 'Unknown').title()} {result.get('model_name', 'Unknown')}")
                
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