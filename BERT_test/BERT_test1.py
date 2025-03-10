from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# GeoBERT 모델 로드
MODEL_NAME = "botryan96/GeoBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, from_tf=True)

# NER 파이프라인 정의
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 예제 문장 실행
text = "The Upper Cambrian fossil was found in Guizhou, China."
ner_results = ner_pipeline(text)

# 결과 출력
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
