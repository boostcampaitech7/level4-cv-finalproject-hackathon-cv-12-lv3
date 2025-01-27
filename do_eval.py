import pandas as pd
from eval import llm_evaluate
from config.config import AI_CONFIG

# CSV 파일 경로
input_csv_path = "output/results.csv"

# CSV 파일 불러오기
results_df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

# 질문, 생성된 답변, 정답 추출
question_list = results_df["question"].tolist()
generated_answers_list = results_df["generated_answer"].tolist()
target_answers_list = results_df["target_answer"].tolist()

# API 키 설정 (config에서 불러오거나 직접 입력)
openai_api_key = AI_CONFIG['openai']
anthropic_api_key = AI_CONFIG['claude']

# 평가 진행
evaluation_results = llm_evaluate(question_list, generated_answers_list, target_answers_list, openai_api_key, anthropic_api_key)

# 평가 결과를 DataFrame에 추가
results_df["evaluation"] = evaluation_results

# 평가 결과를 포함한 CSV 파일 저장
evaluated_csv_path = "output/evaluated_results.csv"
results_df.to_csv(evaluated_csv_path, index=False, encoding="utf-8-sig")
print(f"평가 결과가 포함된 파일이 {evaluated_csv_path}에 저장되었습니다.")

# 평가 결과를 기반으로 통계 계산
total_questions = len(question_list)
total_correct = evaluation_results.count("O")
print(f"전체 질문 수: {total_questions}")
print(f"전체 정답 수: {total_correct}")
print(f"전체 정확도: {(total_correct / total_questions) * 100:.2f}%")