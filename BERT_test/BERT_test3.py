from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# GeoBERT 모델 로드
MODEL_NAME = "botryan96/GeoBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, from_tf=True)

# NER 파이프라인 정의
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# 예제 문장 실행
text = "The Upper Cambrian fossil was found in Guizhou, China."
ner_results = ner_pipeline(text)

# 원본 출력 확인
print("=== Raw NER Output ===")
print(ner_results)

# 결과 출력 (KeyError 방지)
print("\n=== Named Entity Recognition (NER) 결과 ===")
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Confidence: {entity['score']:.4f}")
