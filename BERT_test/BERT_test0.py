from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# GeoBERT 모델 정보
MODEL_NAME = "botryan96/GeoBERT"  # TensorFlow 가중치만 존재

# GeoBERT 토크나이저 및 모델 로드 (TensorFlow weights 사용)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, from_tf=True)

# NER 파이프라인 정의
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 예제 문장
text = """
Innitagnostus ÖPIK, 1967a, p. 98 [*I. innitens; OD;
holotype (ÖPIK, 1967, pl. 58, fig. 2), CPC 5853,
AGSO, Canberra]. Upper Cambrian: China (Guizhou,
Hunan); Australia (Queensland), Mindyallan–Idamean
(E. eretes to S. diloma Zones); Russia (Siberia),
Kazakhstan, G. stolidotus to P. curtare Zones;
Canada (Northwest Territories, British Columbia,
Newfoundland), Glyptagnostus reticulatus to
Olenaspella regularis Zones; USA (Alabama, Nevada,
Texas), Aphelaspis Zone.
"""

# GeoBERT NER 실행
ner_results = ner_pipeline(text)

# 결과 출력
print("=== Named Entity Recognition (NER) 결과 ===")
for entity in ner_results:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}")
