from langchain.document_loaders import PyPDFLoader
from summarizer import Summarizer
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch


class PaperSummarizer:
    def __init__(self):
        self.custom_config = AutoConfig.from_pretrained(
            'allenai/scibert_scivocab_uncased')
        self.custom_config.output_hidden_states = True
        self.custom_tokenizer = AutoTokenizer.from_pretrained(
            'allenai/scibert_scivocab_uncased')
        self.custom_model = AutoModel.from_pretrained(
            'allenai/scibert_scivocab_uncased', config=self.custom_config)

        self.model = Summarizer(
            custom_model=self.custom_model, custom_tokenizer=self.custom_tokenizer)

    def clean_up(self):
        try:
            if hasattr(self, 'custom_model'):
                del self.custom_model
            if hasattr(self, 'model'):
                del self.model

            if 'torch' in globals() and torch is not None:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except:
            pass

    # 자동 메모리 정리
    def __del__(self):
        self.clean_up()

    def generate_summary(self, pdf_filepath):
        try:
            # PyPDFLoader를 사용하여 PDF 파일 로드
            loader = PyPDFLoader(pdf_filepath)
            pages = loader.load()

            # 페이지 정보 저장
            page_dict = {}
            for page in pages:
                page_number = page.metadata.get('page', 'Unknown')
                page_label = page.metadata.get('page_label', 'Unknown')
                page_content = page.page_content.replace('\n', ' ')
                page_dict[page_number] = {
                    'page_label': page_label,
                    'content': page_content
                }

            # 페이지 리스트 생성
            page_list = []
            for page_number, page_info in page_dict.items():
                page_list.append({
                    'page_number': page_number,
                    'page_label': page_info['page_label'],
                    'content': page_info['content']
                })

            # page_label별 요약 수행
            label_summaries = {}
            for page in page_list:
                page_label = page['page_label']
                content = page['content']

                if page_label not in label_summaries:
                    label_summaries[page_label] = ""

                summary = self.model(content, num_sentences=10)
                label_summaries[page_label] += summary + " "

            # 최종 요약 생성
            combined_summary = " ".join(label_summaries.values())
            final_summary = self.model(combined_summary, num_sentences=30)

            return final_summary

        except Exception as e:
            print(f"요약 생성 중 에러 발생: {str(e)}")
            return None
