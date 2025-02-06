import torch
from transformers import AutoModelForCausalLM
from model.deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from model.deepseek_vl.utils.io import load_pil_images
from PIL import Image

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def conversation_with_images(model_path, images, image_description=None, conversation=None, max_new_tokens=512):
    """
    A function to process conversations with images using the DeepSeek VL model.
    
    Args:
        model_path (str): Path to the pre-trained model.
        image_paths (list): List of image file paths (str) to be processed.
        image_description (str, optional): Additional description for the image(s) to be included in the conversation.
        conversation (list, optional): List of conversation dicts, including the user message with image placeholders. 
                                        If None, a default conversation is used.
        max_new_tokens (int): Maximum number of tokens to generate in the response.
        
    Returns:
        str: Generated response from the assistant.
    """
    if not isinstance(images, list):
        images = [images]
        
    # Set default conversation if not provided
    if conversation is None:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>Describe each stage of this image." + (f" {image_description}" if image_description else ""),
                "images": [],
            },
            {"role": "Assistant", "content": ""},
        ]
    
    # Load the pre-trained model and processor
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    # Move the model to GPU with bfloat16 precision
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    # Prepare the conversation and load images (we directly use PIL images here)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=images,  # PIL Image objects directly passed here
        force_batchify=True
    ).to(vl_gpt.device)

    # Get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # Generate the response from the model
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    # Decode the generated response
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    return f"{prepare_inputs['sft_format'][0]} {answer}"


def translate_clova(description, completion_executor):
    messages1 = [
        {
            "role": "system", 
            "content": f"""
            다음은 문서PDF에 들어가는 이미지입니다.
            이미지를 설명하는 문장을 주어진 그대로 자연스럽고 자세하고 더 풍부하게 변환해 주세요. 
            변환된 설명은 이미지의 주요한 요소들을 그대로 반영하면서도, 그 이미지가 떠오를 수 있도록 세밀하고 구체적인 묘사를 추가해 주세요.
            한글로 설명해주어야 합니다.
            여기 주어진 이미지 설명이 있습니다:
            "{description}"
            
            예시:
            "이 이미지는 자연어 처리 작업에 사용되는 신경망 아키텍처인 Transformer의 다이어그램을 보여줍니다. Transformer 아키텍처는 크게 두 가지 주요 부분으로 나뉘며, 왼쪽 부분은 입력 인코딩 단계, 오른쪽 부분은 출력 인코딩 단계를 나타냅니다.
            가장 상단에서 시작되는 입력은 "Inputs"라는 단어로 표시되어 왼쪽 하단에 위치합니다. 이 입력은 네트워크로 전달되어 여러 가지 연산을 통해 네트워크가 처리할 수 있는 형식으로 변환됩니다. 첫 번째 연산은 "Add & Norm"이며, 이는 입력값에 정규화된 값을 더하고 이를 다음 단계로 전달하는 과정입니다. 이 연산은 여러 번 반복되며, 매번 입력값에 정규화된 값을 더하고 이를 다음 단계로 전달합니다.
            이후, 변형된 입력은 Transformer 아키텍처의 핵심 구성 요소인 "Multi-Head Attention" 연산을 거칩니다. 이 연산은 네트워크가 입력의 여러 부분에 동시에 집중할 수 있게 해 주며, 이는 입력의 맥락을 이해하는 데 매우 중요한 역할을 합니다.
            Multi-Head Attention 연산 이후, 변형된 입력은 다시 "Add & Norm" 연산을 거친 후 또 다른 "Multi-Head Attention" 연산을 수행합니다. 이 패턴은 여러 번 반복되며, 각 반복마다 입력값에 정규화된 값을 더하고 이를 다음 단계로 전달합니다.
            마지막으로, 변형된 입력은 "Softmax" 연산을 거쳐 출력값이 1이 되도록 정규화됩니다. 이 연산은 이후 "Linear" 연산을 통해 출력에 선형 변환을 적용하는 과정이 따릅니다."
            """
        }
    ]
  
    request_data1 = {
        'messages': messages1,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 4096,
        'temperature': 0.4,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    res1 = completion_executor.execute(request_data1, stream=False)
    
    res1 = res1['message']['content']
    
    return res1


#### 사용법

# images = ["/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/img/page_4_img_2.png"]
# caption = "This is Transformer Acheitecture img"
# images = [Image.open(image_path) for image_path in images]
# response = conversation_with_images("deepseek-ai/deepseek-vl-7b-chat", images, image_description=caption)
# res = translate_clova(response,completion_executor)
# print(res)