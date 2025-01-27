import json

from api.api_classes import ChatCompletionsExecutor
from config.config import API_CONFIG

prompts ={"first_prompt": """
    Generate a 7-minute podcast conversation based on the provided academic research paper. The conversation should involve two speakers Alex is the podcast MC and an expert on the research paper. Jamie is the guest, asking questions and curious about the research findings.
     1. Structure: Begin with simpler questions to introduce the paper’s basics, gradually moving to more complex and detailed topics. DO NOT switch topics abruptly or start a new episode.
     2. Tone: The conversation should feel natural and engaging. 
    1. Do not introduce new topics abruptly or start a new episode. Maintain continuity in the conversation.
    3. The purpose and motivation behind the research (Why was this study conducted? Why is it important?).
    Write in Korean. Do not conclude the conversation yet. 
    """,

    "second_prompt": """
    Continue the podcast conversation seamlessly from where the first part ended. The same speakers, Alex and Jamie remain in the conversation.

    Guidelines for the conversation flow:
    1. Structure: Begin with a natural transition from the previous discussion (e.g., "Speaking of challenges, what sets this research apart from others?").
        - Highlight the unique contributions and differentiators of the research (e.g., novel findings or innovative methods).
        - Discuss the broader implications or potential applications of the research in real-world scenarios.
    2. Tone: Keep the conversation natural and engaging, as though it’s part of a professional yet relatable podcast.
    3. Dynamics: Alex primarily responds to Jamie’s questions, occasionally offering additional insights or anecdotes

    Critical instructions:
    1. Do not introduce new topics abruptly or start a new episode. Maintain continuity in the conversation.
    2. Avoid adding a closing statement or summary in this part. The conversation should feel like it’s naturally continuing.
    3. Focus of the second part:
        - Unique contributions or differentiators of the research (What makes this study stand out?).
        - Broader implications or applications of the research (How can it be used or applied in the real world?).

    Write in Korean. Ensure the script for this part fills approximately **7 minutes** of spoken content. Do not conclude the conversation yet. 
    """,

    "third_prompt": """
    Continue the podcast conversation seamlessly from where the second part ended. The same speakers, Alex and Jamie , remain in the conversation.

    Guidelines for the conversation flow:
    1. Structure:
        - Dive deeper into the key findings of the research and their significance.
        - Discuss how these findings address gaps or challenges in the field.
        - Provide specific examples or case studies that illustrate the impact of the findings.
        - End with a thoughtful and engaging conclusion, leaving the audience with key takeaways.
    2. Tone:
        - Maintain a conversational and professional tone, with Jamie showing continued curiosity and engagement.
    3. Dynamics: Alex primarily responds to Jamie’s questions, occasionally offering additional insights or anecdotes

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

    # Prepare the messages for the API call
    messages = [
        {"role": "system", "content": prompts[prompt_key]},
        {"role": "user", "content": text_content},
    ]

    if history:
        messages.extend(history)

    # # Calculate and print the token length
    # token_length = calculate_token_length(messages)
    # print(f"[{prompt_key}] Total input token length: {token_length}")

    # Call the API
    response = chat_api.execute(messages,stream=False)
    return response

def write_full_script(text_content):
    """
    Generate the full script by combining all four parts.
    """
    # Generate each part sequentially
    first_response = write_script_part("first_prompt", text_content)
    history = [
        {"role": "system", "content": prompts["first_prompt"]},
        {"role": "assistant", "content": first_response["message"]["content"]},
    ]

    second_response = write_script_part("second_prompt", text_content, history)
    history.append({"role": "system", "content": prompts["second_prompt"]})
    history.append({"role": "assistant", "content": second_response["message"]["content"]})

    third_response = write_script_part("third_prompt", text_content, history)


    # Combine all responses
    full_script = (
        first_response["message"]["content"]
        + "\n\n"
        + second_response["message"]["content"]
        + "\n\n"
        + third_response["message"]["content"]
    )
    return full_script

if __name__ == "__main__":
    text_file_path = "/data/ephemeral/home/script/summary.txt"
    try:
        # Load the text content from file
        with open(text_file_path, "r", encoding="utf-8") as file:
            text_content = file.read()

        # Generate the script
        script = write_full_script(text_content)
        print("Generated Academic Presentation Script:\n")
        print(script)
    except Exception as e:
        print(f"Error occurred: {e}")
