import pandas as pd
import sys
from eval import llm_evaluate
from config.config import AI_CONFIG

# 로그 파일 설정
log_filename = "output/evaluation_log_mlflow.txt"
log_file = open(log_filename, "w", encoding="utf-8")
sys.stdout = log_file  # 표준 출력을 파일로 리디렉션

# CSV 파일 경로
input_csv_path = "/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval_1.CSV"

# CSV 파일 불러오기
results_df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

# 시작 번호 설정
start_index = 166

# 질문, 생성된 답변, 정답 추출 (지정된 인덱스부터)
question_list = results_df["Question"].tolist()[start_index-1:] # 229번부터 시작이므로 228번까지는 제외
generated_answers_list = results_df["Generated Answer"].tolist()[start_index-1:]
target_answers_list = results_df["Target Answer"].tolist()[start_index-1:]

openai_api_key = AI_CONFIG['openai']
anthropic_api_key = AI_CONFIG['claude']

# 평가 진행
evaluation_results = llm_evaluate(question_list, generated_answers_list, target_answers_list, openai_api_key, anthropic_api_key)

# 평가 결과를 DataFrame에 추가 (원래 DataFrame에 추가하기 위해 index 조정)
results_df["evaluation"] = "" # 빈 column 추가
results_df.loc[start_index-1:, "evaluation"] = evaluation_results # 229번 index 부터 evaluation 결과 추가

# 평가 결과를 포함한 CSV 파일 저장
evaluated_csv_path = "output/evaluated_results_ml.csv"
results_df.to_csv(evaluated_csv_path, index=False, encoding="utf-8-sig")
print(f"평가 결과가 포함된 파일이 {evaluated_csv_path}에 저장되었습니다.")

# 평가 결과를 기반으로 통계 계산 (전체 질문 수 및 정확도 다시 계산)
total_questions = len(question_list)
total_correct = evaluation_results.count("O")
print(f"전체 질문 수: {total_questions}")
print(f"전체 정답 수: {total_correct}")
print(f"전체 정확도: {(total_correct / total_questions) * 100:.2f}%")

# 파일 닫기
log_file.close()