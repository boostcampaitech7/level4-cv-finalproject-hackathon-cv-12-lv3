import pandas as pd
from eval import llm_evaluate  # 위의 함수가 포함된 파일이 eval.py에 있다고 가정
from config.config import AI_CONFIG  # API 키 등이 포함된 설정 파일
import logging

# 로그 설정
log_file = "output/evaluation_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# CSV 파일 경로
input_csv_path = "/data/ephemeral/home/lexxsh/level4-cv-finalproject-hackathon-cv-12-lv3/eval_1.CSV"

# CSV 파일 불러오기
results_df = pd.read_csv(input_csv_path, encoding="utf-8-sig")

# 질문, 생성된 답변, 정답 추출
question_list = results_df["Question"].tolist()
generated_answers_list = results_df["Generated_answer"].tolist()
target_answers_list = results_df["Target_answer"].tolist()

# API 키 설정
openai_api_key = AI_CONFIG['openai']
anthropic_api_key = AI_CONFIG['claude']

# 평가 진행
evaluation_results = llm_evaluate(question_list, generated_answers_list, target_answers_list, 
                                  openai_api_key, anthropic_api_key, batch_size=10)

# 평가 결과를 DataFrame에 추가
results_df["evaluation"] = evaluation_results

# 각 문항 번호와 평가 결과 출력
logging.info("\n각 문항별 평가 결과:")
for idx, result in enumerate(evaluation_results, start=1):
    logging.info(f"{idx}번째 문항 평가 결과: {result}")

# 평가 결과를 포함한 CSV 파일 저장
evaluated_csv_path = "output/evaluated_results.csv"
results_df.to_csv(evaluated_csv_path, index=False, encoding="utf-8-sig")
logging.info(f"\n평가 결과가 포함된 파일이 {evaluated_csv_path}에 저장되었습니다.")

# 평가 결과를 기반으로 통계 계산
total_questions = len(question_list)
total_correct = evaluation_results.count("O")
accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
logging.info(f"\n전체 질문 수: {total_questions}")
logging.info(f"전체 정답 수: {total_correct}")
logging.info(f"전체 정확도: {accuracy:.2f}%")

# 종료 메시지
logging.info("평가가 완료되었습니다.")
