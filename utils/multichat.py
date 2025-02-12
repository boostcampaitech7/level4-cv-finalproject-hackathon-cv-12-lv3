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

        base_system_message = {
            "role": "system",
            "content": """
            당신은 학술 논문을 견고하게 설명하는 AI 어시스턴트입니다.

            [핵심 원칙]
            - 상세하고 명확한 설명 (1000~2500 토큰 사이로 답변)
            - 주어진 type 규칙 절대 준수
            - 참조 번호 사용시 논문 목록 필수 포함
            - 친근한 말투 사용 (~이에요, ~네요)

            [type별 규칙]
            1. reference:
            - 제공된 Reference만 사용, 외부 지식 불가
            - "논문에서는~" 형식으로 출처 명시
            - [Page {숫자}] 필수 표기
            - Figure/Table 질문시:
                * 캡션 전체 상세 설명
                * 시각적 요소나 데이터 구조 설명
                * 관련된 본문 내용도 포함

            2. insufficient:
            - 제공 message로 시작
            - 관련 검색 결과만 선별
            - 모든 출처 명확히 표시
            - 비학술적/출처 불명 정보 사용 금지

            3. unrelated/no_result:
            - 시스템 message만 전달
            - 추가 설명 절대 금지
            - 알고 있는 정보도 답변 금지
            - 이전 답변이 이 type이면 추가 설명 요청도 거절

            4. details:
            - 이전 답변 기반 추가 설명
            - 새로운 정보만 제공 (중복 금지)
            - 이전 답변 유형에 따라 검색/논문 기반 답변
            - 주제 이탈 금지

            5. paper_info:
            - 메타데이터만 사용 (제목/저자/년도/초록)
            - "이 논문은 [제목]입니다" 형식 준수

            특히 Figure나 Table 관련 질문에 대해:
            1. 해당 Figure/Table의 내용이 있다면 반드시 설명해주세요
            2. "내용이 없다"고 하기 전에 제공된 내용을 다시 한번 확인해주세요
            3. Figure/Table의 세부 내용과 의미를 자세히 설명해주세요
            
            [실패 조건]
            - Reference 있을 때 외부 지식 사용
            - 비학술적 내용 포함
            - 1000토큰 미만 답변
            - 주제 이탈
            - unrelated/no_result에서 추가 답변

            [details 처리]
            - 이전 답변의 연장으로 처리
            - 이전 type/context 유지
            - 이전이 unrelated/no_result면 설명 거절
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
                "content": f"{context['content']}"
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

        current_tokens = sum(len(msg["content"].split())
                             for msg in self.session_state['preset_messages'])

        available_tokens = 4096 - current_tokens
        max_response_tokens = min(1024, available_tokens - 100)

        return {
            "messages": self.session_state['preset_messages'],
            "maxTokens": max_response_tokens,
            "temperature": 0.3,
            "topK": 0,
            "topP": 0.4,
            "repeatPenalty": 1.3,
            "stopBefore": [],
            "includeAiFilters": True,
            "seed": 0,
        }

    def process_response(self, response):
        """ 응답 처리 및 상태 업데이트 """
        if response:
            try:
                if isinstance(response, str) and 'event:result' in response:
                    result_lines = [line for line in response.split(
                        '\n') if 'event:result' in line or 'data:' in line]
                    for line in result_lines:
                        if 'data:' in line:
                            result_data = json.loads(
                                line.replace('data:', '').strip())
                            if 'message' in result_data:
                                response_text = result_data['message']['content']
                                input_length = result_data.get(
                                    'inputLength', 0)
                                output_length = result_data.get(
                                    'outputLength', 0)

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

                                self.session_state['total_tokens'] = input_length + \
                                    output_length
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

                    self.session_state['total_tokens'] = input_length + \
                        output_length
                    return True

            except Exception as e:
                print(f"응답 처리 중 오류 발생: {str(e)}")
                return False
        return False
