from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

sentence = "BERT is a language model developed by Google."

inputs = tokenizer(sentence, return_tensors="pt", max_length=512, truncation=True)
generated = model.generate(**inputs, max_length=512, num_beams=3)
decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]

print(repr(decoded))