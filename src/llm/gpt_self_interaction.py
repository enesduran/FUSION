import os
import sys
import time
import pprint
import base64
from PIL import Image
from openai import AzureOpenAI
from mimetypes import guess_type

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_utils import create_self_interaction_prompt, create_self_interaction_evaluation_prompt, \
    smplx_vertex_ids, self_interaction_actions


# Azure OpenAI API Configuration
API_BASE = "xxxx"
API_KEY = "xxxx"
DEPLOYMENT_NAME = "xxxx"  
API_VERSION = "2025-01-01-preview"  

# Initialize OpenAI client
def get_openai_client():

    return AzureOpenAI(api_key=API_KEY,
                api_version=API_VERSION,
                base_url=f"{API_BASE}openai/deployments/{DEPLOYMENT_NAME}")

# Convert local image to base64 data URL
def local_image_to_data_url(image_path, max_size=(800, 800)):
    mime_type, _ = guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

# Simple function to ask GPT a question and get a response
def ask_gpt(client, prompt, system_message=None, image_path=None, retries=10):
    """
    Ask GPT a question and get a response.
    
    Args:
        client: OpenAI client instance
        prompt: The question or prompt to send to GPT
        system_message: Optional system message to set the context
        image_path: Optional path to an image to include with the prompt
    
    Returns:
        The response from GPT
    """
    # Default system message if none provided
    if system_message is None:
        system_message = "You are a helpful assistant. Answer concisely."
    
    # Prepare the message content
    content = [{"type": "text", "text": prompt}]
    
    # Add image if provided
    if image_path and os.path.exists(image_path):
        image_content = local_image_to_data_url(image_path)
        content.append({"type": "image_url", "image_url": {"url": image_content}})
    
    # Create the messages for the API call
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": content}]
    

    for attempt in range(retries):

        # Get response from the API
        if 'gpt-4 ' in DEPLOYMENT_NAME:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                max_tokens=2048,
                temperature=0.7)
        else:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=messages,
                max_completion_tokens=2048,
                temperature=1.0) # default value 
        
        message_content = response.choices[0].message.content

        if message_content is None or message_content == '':
            # Log filters and wait before retrying
            print(f"Attempt {attempt+1} failed, retrying...")
            print(response.choices[0].content_filter_results)
            print(response.prompt_filter_results[0]['content_filter_results'])
    
            time.sleep(2 + attempt * 0.5)  # Exponential backoff
            
            pprint.pprint(response.model_dump())

        else:
            return message_content
        
    print("GPT failed to return valid content after retries.")
    return None

# Function to generate command lists for self-interactions
def generate_self_interaction_commands(client, gesture_text):
    """
    Generate command lists for a self-interaction gesture using GPT.
    
    Args:
        client: OpenAI client instance
        gesture_text: Text description of the gesture (e.g., "scratching head")
        
    Returns:
        A dictionary with the gesture text and command list, or None if generation fails
    """
    # Create the prompt for GPT
    prompt = create_self_interaction_prompt(gesture_text)
    
    
    # Set a system message that emphasizes the need for structured output
    system_message = """You are a code generation assistant specialized in producing structured data formats.
    Your task is to generate ONLY the requested Python list structure with no explanations or additional text.
    Do not include markdown code blocks, comments, or any text outside the Python list structure.
    The user needs to parse your response programmatically, so it must be a valid Python list that can be directly evaluated."""
    
    response = ask_gpt(client, prompt, system_message)

    try: 
        # Get response from GPT
        command_list = eval(response)
        return {'text': gesture_text, 'command_list': command_list}
    except:
        print("Failed to parse GPT's response into a valid command list.")
        print("Gesture_text:", gesture_text, "Raw response:", response)
        return None
    

def generate_self_interaction_evaluation_commands(client, command_list):
     
    prompt = create_self_interaction_evaluation_prompt(command_list)

    # Get response from GPT
    response = ask_gpt(client, prompt, '')

    return response

def print_frames_with_vertex_names(frame_data, dict_of_verts):
    
    for spec_instr in frame_data['command_list']:
        
        for jj, stuff in enumerate(spec_instr):
            # stuff [[t1, t2], [[v1, v2], [v3, v4]]]
            
              
            start_frame = stuff[0][0]
            end_frame = stuff[0][1]
            time_range = f"Time frame {start_frame}-{end_frame}:"
            
            for verts_grp in stuff[1]:
                
                group_of_verts = [dict_of_verts[vidx] for vidx in verts_grp]
                print(time_range)
                print('----vertex group:----')
                print('\n'.join(group_of_verts))
                
        print('===================Next interaction with the body.==================')
    



# Example usage
if __name__ == "__main__":
    client = get_openai_client()
    
    # Example: Generate commands for a new gesture
    gesture_text = "touch your left index with your right index finger"  # or any other gesture description
    gestures = ["touch your left index with your right index finger",
                "touch the right thumb with the right index finger",
                "touch your temple with the right hand"]
    
    
    # Generate the command list
    for gest in gestures:
            
        result = generate_self_interaction_commands(client, gest)
    
        smplx_v_ids_inverted = {value: key for key, 
                                    value in smplx_vertex_ids.items()}
        print(f"\n===== Generated Command List for '{gest}' =====")
        
        print_frames_with_vertex_names(result, smplx_v_ids_inverted)
         