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

def llm_evaluate(questions: list, generated_answers: list, target_answers: list, openai_api_key: str, anthropic_api_key: str) -> list:
    """
    여러 평가 방법을 사용하여 LLM이 생성한 답변을 종합 평가합니다.
    LLM 평가와 MLflow 평가를 모두 수행합니다.

    Args:
        questions (list): 질문 목록.
        generated_answers (list): 모델이 생성한 답변 목록.
        target_answers (list): 기준 답변 목록.
        openai_api_key (str): OpenAI API 키.
        anthropic_api_key (str): Anthropic API 키.

    Returns:
        list: 각 답변에 대한 최종 평가 결과 ('O'는 정답, 'X'는 오답).
    """
    # 로거 설정
    logger = logging.getLogger(__name__)

    # API 키 설정
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key

    # 프롬프트 템플릿 정의
    TONIC_ANSWER_SIMILARITY_PROMPT = (
        "Considering the reference answer and the new answer to the following question, "
        "on a scale of 0 to 5, where 5 means the same and 0 means not at all similar, "
        "how similar in meaning is the new answer to the reference answer? Respond with just "
        "a number and no additional text.\nQUESTION: {question}\nREFERENCE ANSWER: {"
        "reference_answer}\nNEW ANSWER: {llm_answer}\n"
    )

    ALLGANIZE_ANSWER_CORRECTNESS_PROMPT = """
    question = \"\"\"
    {question}
    \"\"\"

    target_answer = \"\"\"
    {reference_answer}
    \"\"\"

    generated_answer = \"\"\"
    {llm_answer}
    \"\"\"

    Check if target_answer and generated_answer match by referring to question.
    If target_answer and generated_answer match 1, answer 0 if they do not match.
    Only 1 or 0 must be created.
    """

    # TONIC 유사성 검증
    def tonic_validate(questions: list, generated_answers: list, target_answers: list, model: str) -> list:
        llm = ChatOpenAI(model_name=model)
        prompt = PromptTemplate(
            input_variables=["question", "reference_answer", "llm_answer"],
            template=TONIC_ANSWER_SIMILARITY_PROMPT,
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        eval_check = []
        for question, target_answer, generated_answer in zip(tqdm(questions), target_answers, generated_answers):
            try:
                llm_result = chain.run(
                    {"question": question, "reference_answer": target_answer, "llm_answer": generated_answer}
                )
                eval_check.append(int(llm_result))
            except Exception as e:
                logger.warning(f"llm_eval exception: {e}")
                eval_check.append(-1)
        return eval_check

    # ALLGANIZE 정확성 검증
    def allganize_eval(questions: list, generated_answers: list, target_answers: list, model: str) -> list:
        llm = ChatAnthropic(model=model)
        prompt = PromptTemplate(
            input_variables=["question", "reference_answer", "llm_answer"],
            template=ALLGANIZE_ANSWER_CORRECTNESS_PROMPT,
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        eval_check = []
        for question, target_answer, generated_answer in zip(tqdm(questions), target_answers, generated_answers):
            try:
                llm_result = chain.run(
                    {"question": question, "reference_answer": target_answer, "llm_answer": generated_answer}
                )
                eval_check.append(int(llm_result))
            except Exception as e:
                logger.warning(f"llm_eval exception: {e}")
                eval_check.append(-1)
        return eval_check

    # MLflow 평가
    def mlflow_eval(question_list: list, answer_list: list, ground_truth_list: list, model: str):
        eval_data = pd.DataFrame({"inputs": question_list, "predictions": answer_list, "ground_truth": ground_truth_list})

        with mlflow.start_run():
            # MLflow 평가 실행
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
            mlflow_answer_similarity = eval_table["answer_similarity/v1/score"].tolist()
            mlflow_answer_correctness = eval_table["answer_correctness/v1/score"].tolist()

        return mlflow_answer_similarity, mlflow_answer_correctness

    # 가장 빈번한 요소 반환
    def most_frequent_element(result: list) -> str:
        count = Counter(result)
        priority = ["X", "O"]
        most_common = count.most_common()
        for element in priority:
            if element in count and count[element] == most_common[0][1]:
                return element

    # 평가 점수를 'O' 또는 'X'로 분류
    def get_evaluation_result(score: int) -> str:
        return "O" if score >= 4 else "X"

    # 평가 결과 종합
    def eval_vote(
        tonic_answer_similarity: list,
        mlflow_answer_similarity: list,
        mlflow_answer_correctness: list,
        allganize_answer_correctness: list,
    ) -> list:
        e2e_result = []
        for i in range(len(tonic_answer_similarity)):
            tonic_answer_similarity_ox = get_evaluation_result(tonic_answer_similarity[i])
            mlflow_answer_similarity_ox = get_evaluation_result(mlflow_answer_similarity[i])
            mlflow_answer_correctness_ox = get_evaluation_result(mlflow_answer_correctness[i])
            allganize_answer_correctness_ox = "O" if allganize_answer_correctness[i] == 1 else "X"

            e2e_result.append(
                most_frequent_element(
                    [
                        tonic_answer_similarity_ox,
                        mlflow_answer_similarity_ox,
                        mlflow_answer_correctness_ox,
                        allganize_answer_correctness_ox,
                    ]
                )
            )
        return e2e_result

    # 평가 실행
    tonic_answer_similarity = tonic_validate(questions, generated_answers, target_answers, model="gpt-3.5-turbo")
    allganize_answer_correctness = allganize_eval(questions, generated_answers, target_answers, model="claude-3-opus-20240229")
    mlflow_answer_similarity, mlflow_answer_correctness = mlflow_eval(questions, generated_answers, target_answers, model="openai:/gpt-3.5-turbo")

    # 각 평가 결과 출력
    print("TONIC 유사성 검증 결과:", tonic_answer_similarity)
    print("MLflow 유사성 검증 결과:", mlflow_answer_similarity)
    print("MLflow 정확성 검증 결과:", mlflow_answer_correctness)
    print("ALLGANIZE 정확성 검증 결과:", allganize_answer_correctness)

    # 결과 종합
    final_results = eval_vote(tonic_answer_similarity, mlflow_answer_similarity, mlflow_answer_correctness, allganize_answer_correctness)
    print("최종 평가 결과:", final_results)

    return final_results