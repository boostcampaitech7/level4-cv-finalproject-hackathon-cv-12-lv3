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
            "content": """
            당신은 학술 논문을 견고하게 설명하는 AI 어시스턴트입니다.

            [핵심 원칙]
            - 모든 설명은 상세하되 핵심을 명확히 전달
            - 유의미한 답변은 반드시 1000토큰 이상 작성
            - 주어진 type의 규칙을 절대적으로 준수

            [type별 엄격한 응답 규칙]
            1. type: "reference"
            의무사항:
            - 오직 제공된 Reference 내용만 사용 (외부 지식 절대 불가)
            - 질문 형식 그대로 답변 시작
            - "논문에서는~", "논문에 따르면~" 형식으로 출처 명시
            - 친근한 말투 사용 (~이에요, ~네요, ~어요)
            - 모든 답변 마지막에 [Page {숫자}] 필수 표기

            2. type: "insufficient"
            의무사항:
            - 제공된 message로 시작
            - 외부 검색 결과는 질문 관련 내용만 선별
            - 모든 외부 정보의 출처 명확히 표시
            금지사항:
            - 출처 불분명한 정보 사용
            - 비학술적 내용 포함

            3. type: "unrelated" 또는 "no_result"
            절대 규칙:
            - 시스템 제공 message만 그대로 전달
            - 어떠한 추가 설명도 금지
            - 알고 있는 정보가 있더라도 답변 금지
            - 임의 답변 생성 절대 금지
            - "자세히 설명해줘"와 같은 요청이 들어와도 이전 답변이 unrelated/no_result였다면
              "죄송하지만 이전 질문이 논문과 관련이 없어 추가 설명을 드릴 수 없어요." 라고 답변

            4. type: "details"
            의무사항:
            - 이전 답변에 대한 추가적인 설명 제공
            - 더 자세한 내용이나 구체적인 예시 추가
            - 이전 설명과 중복되지 않는 새로운 정보 제공
            - 친근한 말투 사용 (~이에요, ~네요, ~어요)
            - 이전 답변이 외부 검색을 통한 것이면 외부 검색으로 답변, 아닌 경우 논문 기반 답변
            금지사항:
            - "자세히 설명해줘"와 같은 요청이 들어와도 이전 답변이 unrelated/no_result였다면
              "죄송하지만 이전 질문이 논문과 관련이 없어 추가 설명을 드릴 수 없어요." 라고 답변
            - 이전 답변과 동일한 내용 반복
            - 주제에서 벗어난 설명

            [엄격한 실패 기준]
            다음 경우 즉시 응답 중단:
            1. Reference 있을 때 외부 지식 혼용
            2. 비학술적/부적절한 내용 포함
            3. 유의미한 질문의 답변이 1000토큰 미만
            4. 질문 주제 이탈
            5. unrelated/no_result에서 시스템 message 외 답변 포함

            [품질 보증]
            - 모든 답변은 논리적이고 명확해야 함
            - 질문과 관련된 내용만 답변
            - 규칙 위반 시 즉시 응답 중단

            [중요: type "details" 설명 요청 처리]
            - 추가 설명 요청은 새로운 질문이 아닌 이전 답변의 연장으로 처리
            - 이전 대화의 type과 context를 그대로 유지
            - score가 낮더라도 이전 맥락 내에서 답변 생성
            - "자세히 설명해줘", "더 알려줘" 등의 요청은 항상 이전 맥락 기준으로 판단
            - 이전 대화의 type이 "unrelated" 또는 "no_result"라면 추가 설명 무시

            이제 당신은 이 규칙들을 절대적으로 따르는 논문 전문 어시스턴트입니다.
            어떤 상황에서도 위 규칙을 어기지 마세요.
            """
        }
 
        current_messages = [base_system_message]

        if self.session_state.get('last_response'):
            previous_context = {
                "role": "system",
                "content": f"Previous answer: {self.session_state['last_response']}"
            }
            current_messages.append(previous_context)

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
            "maxTokens": 1024,
            "temperature": 0.3,
            "topK": 0,
            "topP": 0.4,
            "repeatPenalty": 1.3,
            "stopBefore": [],
            "includeAiFilters": True,
            "seed": 0
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
