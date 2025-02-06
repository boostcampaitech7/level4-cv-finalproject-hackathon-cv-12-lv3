import logging
from collections import Counter
from tqdm import tqdm
import mlflow
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
from typing import List, Tuple, Dict

def llm_evaluate(questions: list, generated_answers: list, target_answers: list, 
                openai_api_key: str, anthropic_api_key: str, batch_size: int = 5) -> list:
    """
    여러 평가 방법을 사용하여 LLM이 생성한 답변을 종합 평가합니다.
    LLM 평가와 MLflow 평가를 모두 수행합니다.

    Args:
        questions (list): 질문 목록
        generated_answers (list): 모델이 생성한 답변 목록
        target_answers (list): 기준 답변 목록
        openai_api_key (str): OpenAI API 키
        anthropic_api_key (str): Anthropic API 키
        batch_size (int): 배치 크기 (기본값: 5)

    Returns:
        list: 각 답변에 대한 최종 평가 결과 ('O'는 정답, 'X'는 오답)
    """
    # 로거 설정
    logger = logging.getLogger(__name__)

    # API 키 설정
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # 프롬프트 템플릿 정의
    BATCH_TONIC_PROMPT = """
    For each answer pair below, rate the semantic similarity on a scale of 0 to 5, 
    where 5 means identical meaning and 0 means completely different.
    Provide only the numerical score for each pair, one score per line.

    {batch_content}
    """

    BATCH_ALLGANIZE_PROMPT = """
    For each answer pair below, determine if they match semantically.
    Respond with exactly 1 for a match or 0 for a mismatch.
    Focus on semantic meaning rather than exact wording.
    Provide only the binary score (0 or 1) for each pair, one score per line.

    {batch_content}
    """

    EVAL_TEMPLATE = """
    Question: {question}
    Reference Answer: {reference_answer}
    Generated Answer: {llm_answer}
    """

    def create_batch_prompts(questions: list, target_answers: list, 
                           generated_answers: list, batch_size: int) -> list:
        """배치 단위로 프롬프트를 생성합니다."""
        batched_prompts = []
        for i in range(0, len(questions), batch_size):
            batch_content = ""
            for j in range(i, min(i + batch_size, len(questions))):
                batch_content += f"\n--- Evaluation #{j+1} ---\n"
                batch_content += EVAL_TEMPLATE.format(
                    question=questions[j],
                    reference_answer=target_answers[j],
                    llm_answer=generated_answers[j]
                )
            batched_prompts.append(batch_content)
        return batched_prompts

    def parse_batch_response(response: str, batch_size: int) -> list:
        """배치 응답을 개별 점수로 파싱합니다."""
        scores = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            try:
                if line and (line.replace('.', '').isdigit() or line in ['0', '1']):
                    scores.append(float(line))
            except ValueError:
                continue
        
        # 배치 크기만큼 결과가 없는 경우 -1로 채움
        if len(scores) < batch_size:
            scores.extend([-1] * (batch_size - len(scores)))
        return scores[:batch_size]

    # TONIC 유사성 검증 (배치 처리)
    def tonic_validate(questions: list, generated_answers: list, 
                      target_answers: list, model: str, batch_size: int) -> list:
        llm = ChatOpenAI(model_name=model)
        batched_prompts = create_batch_prompts(questions, target_answers, 
                                             generated_answers, batch_size)
        
        eval_check = []
        for batch_prompt in tqdm(batched_prompts, desc="TONIC Evaluation"):
            try:
                result = llm.invoke(BATCH_TONIC_PROMPT.format(batch_content=batch_prompt))
                batch_scores = parse_batch_response(result.content, batch_size)
                eval_check.extend(batch_scores)
            except Exception as e:
                logger.warning(f"TONIC evaluation exception: {e}")
                eval_check.extend([-1] * batch_size)
        return eval_check

    # ALLGANIZE 정확성 검증 (배치 처리)
    def allganize_eval(questions: list, generated_answers: list, 
                      target_answers: list, model: str, batch_size: int) -> list:
        llm = ChatAnthropic(model=model)
        batched_prompts = create_batch_prompts(questions, target_answers, 
                                             generated_answers, batch_size)
        
        eval_check = []
        for batch_prompt in tqdm(batched_prompts, desc="ALLGANIZE Evaluation"):
            try:
                result = llm.invoke(BATCH_ALLGANIZE_PROMPT.format(batch_content=batch_prompt))
                batch_scores = parse_batch_response(result.content, batch_size)
                eval_check.extend(batch_scores)
            except Exception as e:
                logger.warning(f"ALLGANIZE evaluation exception: {e}")
                eval_check.extend([-1] * batch_size)
        return eval_check

    # MLflow 평가 (배치 처리)
    def mlflow_eval(question_list: list, answer_list: list, 
                   ground_truth_list: list, model: str, batch_size: int) -> Tuple[list, list]:
        similarities = []
        correctness = []
        
        for i in range(0, len(question_list), batch_size):
            batch_questions = question_list[i:i + batch_size]
            batch_answers = answer_list[i:i + batch_size]
            batch_truths = ground_truth_list[i:i + batch_size]
            
            eval_data = pd.DataFrame({
                "inputs": batch_questions,
                "predictions": batch_answers,
                "ground_truth": batch_truths
            })
            
            try:
                with mlflow.start_run(nested=True):
                    results = mlflow.evaluate(
                        data=eval_data,
                        targets="ground_truth",
                        predictions="predictions",
                        extra_metrics=[
                            mlflow.metrics.genai.answer_similarity(model=model),
                            mlflow.metrics.genai.answer_correctness(model=model),
                        ],
                        evaluators="default",
                    )
                    
                    eval_table = results.tables["eval_results_table"]
                    batch_similarities = eval_table["answer_similarity/v1/score"].tolist()
                    batch_correctness = eval_table["answer_correctness/v1/score"].tolist()
                    
                    similarities.extend(batch_similarities)
                    correctness.extend(batch_correctness)
            except Exception as e:
                logger.warning(f"MLflow evaluation exception: {e}")
                similarities.extend([-1] * len(batch_questions))
                correctness.extend([-1] * len(batch_questions))
        
        return similarities, correctness

    # 가장 빈번한 요소 반환
    def most_frequent_element(result: list) -> str:
        count = Counter(result)
        priority = ["X", "O"]  # X가 우선순위가 높음 (보수적 평가)
        most_common = count.most_common()
        for element in priority:
            if element in count and count[element] == most_common[0][1]:
                return element
        return most_common[0][0]

    # 평가 점수를 'O' 또는 'X'로 분류
    def get_evaluation_result(score: float) -> str:
        if score < 0:  # 에러 케이스
            return 'X'
        return 'O' if score >= 4 else 'X'

    # 평가 실행
    tonic_answer_similarity = tonic_validate(
        questions, generated_answers, target_answers, 
        model="gpt-3.5-turbo", batch_size=batch_size
    )
    
    allganize_answer_correctness = allganize_eval(
        questions, generated_answers, target_answers, 
        model="claude-3-opus-20240229", batch_size=batch_size
    )
    
    mlflow_answer_similarity, mlflow_answer_correctness = mlflow_eval(
        questions, generated_answers, target_answers,
        model="openai:/gpt-3.5-turbo", batch_size=batch_size
    )

    # 각 평가 결과 출력
    print("TONIC 유사성 검증 결과:", tonic_answer_similarity)
    print("MLflow 유사성 검증 결과:", mlflow_answer_similarity)
    print("MLflow 정확성 검증 결과:", mlflow_answer_correctness)
    print("ALLGANIZE 정확성 검증 결과:", allganize_answer_correctness)

    # 각 평가를 O/X로 변환
    tonic_results = [get_evaluation_result(score) for score in tonic_answer_similarity]
    mlflow_sim_results = [get_evaluation_result(score) for score in mlflow_answer_similarity]
    mlflow_corr_results = [get_evaluation_result(score) for score in mlflow_answer_correctness]
    allganize_results = ['O' if score == 1 else 'X' for score in allganize_answer_correctness]

    # 결과 종합
    final_results = []
    for i in range(len(tonic_results)):
        result = most_frequent_element([
            tonic_results[i],
            mlflow_sim_results[i],
            mlflow_corr_results[i],
            allganize_results[i]
        ])
        final_results.append(result)

    print("최종 평가 결과:", final_results)
    return final_results