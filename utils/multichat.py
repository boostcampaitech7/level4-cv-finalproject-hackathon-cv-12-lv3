import json

class MultiChatManager:
    def __init__(self):
        self.session_state = {
            "preset_messages": [],
            "total_tokens": 0,
            "chat_log": [],
            "started": False,
            "summary_messages": [],
            "last_user_input": "",
            "last_response": "",
            "last_user_message": {},
            "last_assistant_message": {},
            "previous_messages": [],
            "system_message_changed": False,
            "system_message": ""
        }

    def initialize_chat(self, system_message):
        """ 채팅 초기화 및 시스템 메시지 설정 """
        self.session_state['system_message'] = system_message
        self.session_state['system_message_changed'] = True
        self.session_state['started'] = True

        if self.session_state['system_message_changed']:
            self.session_state['preset_messages'] = [
                msg for msg in self.session_state['preset_messages']
                if msg['role'] != 'system'
            ]
            self.session_state['preset_messages'].insert(
                0,
                {"role": "system", "content": system_message}
            )
            self.session_state['system_message_changed'] = False

    def prepare_chat_request(self, user_input, context=None):
        """ 채팅 요청 데이터 준비 """
        self.session_state['last_user_input'] = user_input

        # 프롬프트 설정
        base_system_message = {
            "role": "system",
            "content": "\n".join([
                "You are an AI assistant specialized in explaining concepts from academic papers.",
                "Include the page number from the reference in your answer.",
                "Base your answer solely on the provided reference.",
                "Keep your explanation clear and concise."
            ])
        }

        current_messages = [base_system_message]

        if context:
            context_message = {
                "role": "system",
                "content": f"Reference documents:\n{context}"
            }
            current_messages.append(context_message)

        user_message = {
            "role": "user",
            "content": user_input
        }
        current_messages.append(user_message)

        self.session_state['last_user_message'] = user_message
        self.session_state['preset_messages'] = current_messages
        self.session_state['chat_log'].append(user_message)

        return {
            "messages": self.session_state['preset_messages'],
            "maxTokens": 256,
            "temperature": 0.5,
            "topK": 0,
            "topP": 0.6,
            "repeatPenalty": 1.2,
            "stopBefore": [],
            "includeAiFilters": True,
            "seed": 0,
        }
    
    def process_response(self, response):
        """ 응답 처리 및 상태 업데이트 """
        if response:
            try:
                if isinstance(response, str) and 'event:result' in response:
                    result_lines = [line for line in response.split('\n') if 'event:result' in line or 'data:' in line]
                    for line in result_lines:
                        if 'data:' in line:
                            result_data = json.loads(line.replace('data:', '').strip())
                            if 'message' in result_data:
                                response_text = result_data['message']['content']
                                input_length = result_data.get('inputLength', 0)
                                output_length = result_data.get('outputLength', 0)
                                
                                self.session_state['last_response'] = response_text
                                self.session_state['last_assistant_message'] = {
                                    "role": "assistant",
                                    "content": response_text
                                }

                                self.session_state['preset_messages'].append(
                                    self.session_state['last_assistant_message']
                                )
                                self.session_state['chat_log'].append(
                                    self.session_state['last_assistant_message']
                                )

                                self.session_state['total_tokens'] = input_length + output_length
                                return True
                else:
                    response_text = response['context']
                    input_length = response.get('inputLength', 0)
                    output_length = response.get('outputLength', 0)
                    
                    self.session_state['last_response'] = response_text
                    self.session_state['last_assistant_message'] = {
                        "role": "assistant",
                        "content": response_text
                    }

                    self.session_state['preset_messages'].append(
                        self.session_state['last_assistant_message']
                    )
                    self.session_state['chat_log'].append(
                        self.session_state['last_assistant_message']
                    )

                    self.session_state['total_tokens'] = input_length + output_length
                    return True
                    
            except Exception as e:
                print(f"응답 처리 중 오류 발생: {str(e)}")
                return False
        return False
    
    def check_token_limit(self, max_tokens):
        """ 토큰 제한 체크"""
        token_limit = 4096 - max_tokens
        return self.session_state['total_tokens'] > token_limit
