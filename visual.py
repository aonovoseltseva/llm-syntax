import os stanza re argparse torch

from transformers import TextStreamer AutoTokenizer BitsAndBytesConfig
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

stanza.download('ru')  

# NLP-пайплайн
nlp = stanza.Pipeline(lang='ru', processors='tokenize,mwt,pos,lemma,depparse')
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

system_prompt = """You are working on the code based visualising of syntactic structures of Russian sentences.
                You will be given a sentence, and you should return its dependency relations in the code format"""

def process_chunks_with_dependency_parse(chunks_dir='chunk_folder', 
                                       results_path='results.txt',
                                       code_output_dir='visualization_code'):
    # Создаем папку для сохранения кода визуализации
    os.makedirs(code_output_dir, exist_ok=True)
    
    streamer = TextStreamer(tokenizer)  # Опционально: для потокового вывода
    
    with open(results_path, 'w', encoding='utf-8') as out_file:
        for filename in sorted(os.listdir(chunks_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(chunks_dir, filename)
                out_file.write(f"\n===== Обработка файла: {filename} =====\n")
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        sentence = line.strip()
                        if not sentence:
                            continue
                            
                        out_file.write(f"\n▶ Предложение {i+1}: {sentence}\n")
                        doc = nlp(sentence)
                        out_file.write("Dependency Relations:\n")
                        
                        # Формируем зависимости для промпта
                        dep_lines = []
                        for sent in doc.sentences:
                            for word in sent.words:
                                head = sent.words[word.head - 1].text if word.head > 0 else 'ROOT'
                                dep_line = f"{word.text} ({word.deprel}) -> {head}"
                                dep_lines.append(dep_line)
                                out_file.write(dep_line + "\n")
                        
                        # Создаем промпт
                        dep_prompt = "\n".join(dep_lines)
                        prompt_text = (
                            f"Here is a syntactic structure:\n{dep_prompt}\n"
                            "Write a complete Python code to visualize it using "
                            "Matplotlib, NetworkX, or PyVis. Include all necessary imports. "
                            "The code should be ready to run in a Jupyter notebook."
                        )
                        
                        messages = [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": prompt_text}]
                            },
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": "Output only the Python code wrapped in ```python ``` blocks."}]
                            },
                        ]
                        
                        # Генерируем код
                        inputs = tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            return_tensors="pt",
                        ).to(model.device)
                        
                        with torch.inference_mode():
                            outputs = model.generate(**inputs, max_new_tokens=1500)
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Извлекаем код из ответа
                        code_blocks = re.findall(r'```python(.*?)```', generated_text, re.DOTALL)
                        
                        if code_blocks:
                            # Сохраняем код в отдельный файл
                            code_filename = f"{filename[:-4]}_sentence_{i+1}.py"
                            code_path = os.path.join(code_output_dir, code_filename)
                            
                            with open(code_path, 'w', encoding='utf-8') as code_file:
                                code_file.write(code_blocks[0].strip())
                            
                            out_file.write(f"\nКод визуализации сохранен в: {code_path}\n")
                            out_file.write("Содержимое кода:\n")
                            out_file.write(code_blocks[0].strip() + "\n")
                            
                            # Возвращаем путь к файлу с кодом
                            yield code_path
                        else:
                            out_file.write("\n⚠ Не удалось извлечь код из ответа модели\n")
                            yield None