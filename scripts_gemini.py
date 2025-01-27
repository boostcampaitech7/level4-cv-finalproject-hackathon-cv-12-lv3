import json
import google.generativeai as genai

# Configuration for the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# API 키 설정
genai.configure(api_key="AIzaSyCG9Kd6Cc0DxR8-nu0m8uV3vqtzDrkrRFo")


write_script_config = {
    "model_name": "gemini-1.5-flash-002",
    "generation_config": generation_config,
}

prompts = {
    "first_prompt": """Generate a 7-minute podcast conversation based on the provided academic research paper. The conversation should involve two speakers. '재석' is the podcast MC and an expert on the research paper. '하하' is the guest, asking questions and curious about the research findings.
Ensure the output is always a dictionary with the key 'conversations' containing a list of dictionaries. Each dictionary in the list must include the keys '재석' and '하하하'. Do not use any other structure or format.

Guidelines for the conversation flow:
1. Structure: Begin with simpler questions to introduce the paper’s basics, gradually moving to more complex and detailed topics.
2. Tone: The conversation should feel natural and engaging.
3. Dynamics: Alex primarily responds to Jamie’s questions, occasionally offering additional insights or anecdotes to make the conversation lively and relatable.

Additional context:
- Write in Korean
- The first half should contain 15 pairs of conversations, making it richer and more detailed.
- Welcome the listeners with a super fun overview and maintain an approachable tone to appeal to a wide audience
- DO NOT finish the entire conversation in this section. Do not include any conclusions or summaries at the end of this section.

First, show me the first 15 pairs of conversations.
DO NOT finish the conversation just yet.
""",

 "second_prompt": """Now, show me the remaining part of the 10-minute podcast conversation.
The second half should also consist of 15 pairs of conversations.

Don't forget to conclude with a brief summary or takeaway that reflects the research’s impact or next steps in the field.
Also, DO NOT say anything like 'please subscribe to our channel' or 'please like and share' or anything like that. Write in Korean
""",
}

def read_pdf_as_text(file_path):
    """
    Reads a text file containing the academic research paper.
    Args:
        file_path (str): Path to the text file.
    Returns:
        str: The content of the text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def write_script_first_half(pdf_text):
    """
    Generate the first half of the podcast script.
    Args:
        pdf_text (str): The text content of the academic research paper.
    Returns:
        tuple: Chat history and the first half of the script.
    """
    model = genai.GenerativeModel(
        model_name=write_script_config["model_name"],
        generation_config=write_script_config["generation_config"],
    )

    chat_session = model.start_chat(
        history=[
            {"role": "user", "parts": [pdf_text]},
        ]
    )

    prompt = prompts["first_prompt"]
    response = chat_session.send_message(prompt)

    history = [
        {"role": "user", "parts": [prompt]},
        {"role": "assistant", "parts": [response.text]},
    ]

    return history, json.loads(response.text)

def write_script_second_half(history):
    """
    Generate the second half of the podcast script.
    Args:
        history (list): The chat history from the first half.
    Returns:
        dict: The second half of the script.
    """
    model = genai.GenerativeModel(
        model_name=write_script_config["model_name"],
        generation_config=write_script_config["generation_config"],
    )

    chat_session = model.start_chat(history=history)

    prompt = prompts["second_prompt"]
    response = chat_session.send_message(prompt)

    return json.loads(response.text)

def write_script(file_path):
    """
    Generate the full podcast script based on the academic research paper.
    Args:
        file_path (str): Path to the text file containing the academic research paper.
    """
    pdf_text = read_pdf_as_text(file_path)

    # Generate first half of the script
    history, first_half_script = write_script_first_half(pdf_text)

    # Generate second half of the script
    second_half_script = write_script_second_half(history)
    # print("First Half Script:", first_half_script)
    # print("Second Half Script:", second_half_script)

    
    #full_script = first_half_script + second_half_script
    full_script = first_half_script["conversations"] + second_half_script["conversations"]


    # Print the full script to the terminal
    print(json.dumps(full_script, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    text_file_path = "/data/ephemeral/home/script/summary.txt" # Path to your text file
    write_script(text_file_path)
