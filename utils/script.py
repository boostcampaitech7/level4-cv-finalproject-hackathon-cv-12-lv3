import json
from api.api_classes import ChatCompletionsExecutor
from config.config import API_CONFIG

prompts = {
    "first_prompt": """
    Create the first part of a podcast script based on an academic research paper. 
    Speakers: 재석 (host, expert on the paper), 하하 (guest, curious).

    Focus:
    1. Purpose and motivation of the research (Why it was conducted and its importance).
    2. Research methods and unique approach.

    Style:
    - Natural and engaging conversation.
    - Gradually build complexity without abrupt transitions.
    - End with 하하 asking a question to continue the discussion 
    **Output format:**
    재석: [대화 내용]
    하하: [대화 내용]

    **Do not conclude the conversation.**
    Write in Korean.
    """,

    "second_prompt": """
    Continue the podcast seamlessly from the first part.
    Speakers: 재석 (host), 하하 (guest).

    Focus:
    1. Unique contributions of the research (What makes it stand out?).
    2. Broader implications or real-world applications.

    Style:
    - Natural flow from the previous part.
    - Keep the tone conversational and professional.
    - End with 하하 asking an open-ended question to transition to the next part 
    **Output format:**
    재석: [대화 내용]
    하하: [대화 내용]

    **Do not conclude the conversation.**
    Write in Korean.
    """,

    "third_prompt": """
    Continue the podcast seamlessly from the second part.
    Speakers: 재석 (host), 하하 (guest).

    Focus:
    1. Key findings and their significance.
    2. Examples or case studies illustrating the findings' impact.
    3. Future directions or challenges the research addresses.

    Style:
    - Smooth continuation from the previous discussion.
    - Conclude thoughtfully with key takeaways 
    - Closing remarks by 재석 to end the podcast
    **Output format:**
    재석: [대화 내용]
    하하: [대화 내용]

    Write in Korean.
    """
}

def write_script_part(prompt_key, text_content, history=None):
    """
    Generate a part of the script based on the specified prompt.
    """
    chat_api = ChatCompletionsExecutor(
        host=API_CONFIG["host"],
        api_key=API_CONFIG["api_key"],
        request_id=API_CONFIG["request_id"],
    )
    
    messages = [
        {"role": "system", "content": prompts[prompt_key]},
        {"role": "user", "content": text_content},
    ]
    if history:
        messages.extend(history)

    request_data = {
        'messages': messages,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 4096,
        'temperature': 0.6,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 1234
    }
    response = chat_api.execute(request_data, stream=False)
    return response

def write_full_script(text_content):
    """
    Generate the full script by combining all four parts.
    """
    first_response = write_script_part("first_prompt", text_content)
    history = [
        {"role": "system", "content": prompts["first_prompt"]},
        {"role": "assistant", "content": first_response["message"]["content"]},
    ]

    second_response = write_script_part("second_prompt", text_content, history)
    history.append({"role": "system", "content": prompts["second_prompt"]})
    history.append({"role": "assistant", "content": second_response["message"]["content"]})

    third_response = write_script_part("third_prompt", text_content, history)

    full_script = (
        first_response["message"]["content"]
        + "\n\n"
        + second_response["message"]["content"]
        + "\n\n"
        + third_response["message"]["content"]
    )      
    # Parse the script into JSON format for TTS compatibility
    conversations = []
    lines = full_script.strip().split("\n")
    current_conversation = {"재석": "", "하하": ""}

    for line in lines:
        if line.startswith("재석 :"):
            current_conversation["재석"] = line[len("재석 :"):].strip()
        elif line.startswith("하하 :"):
            current_conversation["하하"] = line[len("하하 :"):].strip()
            conversations.append(current_conversation)
            current_conversation = {"재석": "", "하하": ""}

    if current_conversation["재석"] or current_conversation["하하"]:
        conversations.append(current_conversation)

    return conversations
