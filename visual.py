import os
import stanza
import argparse
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import GemmaForCausalLM as Gemma3ForCausalLM  

parser = argparse.ArgumentParser(description="Инструмент для парсинга синтаксической структуры")
print("parser started")
parser.add_argument("--database", type=str, required=True, help="Путь к файлу с трибанком в формате CoNLL-U")
parser.add_argument("--results", type=str, required=True, help="Путь к файлу для сохранения результатов")
parser.add_argument('--quantization', type=lambda x: x.lower() == 'true', default=False)
args = parser.parse_args()
print("parser ended")

print("second block started")
quantization = args.quantization
database_path = args.database
output_results = args.results
model_id = "google/gemma-3-1b-it"

# Применяем квантизацию: загружаем модель меньшей размерности
quantization_config = BitsAndBytesConfig(load_in_8bit=True) if args.quantization == True else None

# Инициализация модели из HuggingFace: загружается локально на наше устройство
# Это значит, что она не использует сторонние сервисы, а все вычисления выполняются у нас
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

# Токенизация тоже производится локально, т.е. на нашем устройстве
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Загрузи модель один раз (например, для английского)
stanza.download('en')  # для русского: 'ru'

# Загружаем NLP-пайплайн
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

# Функция dependency-парсинга
def dependency_parse(sentence):
    doc = nlp(sentence)
    for sent in doc.sentences:
        print("Dependency Relations:")
        for word in sent.words:
            print(f"{word.text}\t({word.deprel})\t->\t{sent.words[word.head - 1].text if word.head > 0 else 'ROOT'}")
def chunks_from_conllu(database_path, chunk_size=100, output_prefix='chunk', output_dir='chunk_folder'):
    os.makedirs(output_dir, exist_ok=True) 

    texts = []
    chunk_index = 0
    count = 0
    chunk_files = []

    with open(database_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('# text ='):
                sentence = line[len('# text ='):].strip()
                texts.append(sentence)
                count += 1

                if count % chunk_size == 0:
                    chunk_file = os.path.join(output_dir, f"{output_prefix}_{chunk_index}.txt")
                    with open(chunk_file, 'w', encoding='utf-8') as out:
                        out.write('\n'.join(texts))
                    chunk_files.append(chunk_file)
                    texts = []
                    chunk_index += 1

        # Сохраняем оставшиеся предложения
        if texts:
            chunk_file = os.path.join(output_dir, f"{output_prefix}_{chunk_index}.txt")
            with open(chunk_file, 'w', encoding='utf-8') as out:
                out.write('\n'.join(texts))
            chunk_files.append(chunk_file)

    print(f"     Created {len(chunk_files)} chunk files in '{output_dir}' from {count} sentences.")


