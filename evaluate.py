import argparse, re, random, os
import torch
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM




parser = argparse.ArgumentParser(description="Инструмент для парсинга синтаксической структуры")
print("parser started")
parser.add_argument("--prompts", type=str, required=True, help="Путь к файлу с тестовыми данными")
parser.add_argument("--database", type=str, required=True, help="Путь к файлу с трибанком в формате CoNLL-U")
parser.add_argument("--results", type=str, required=True, help="Путь к файлу для сохранения результатов")
parser.add_argument('--quantization', type=lambda x: x.lower() == 'true', default=False)
args = parser.parse_args()
print("parser ended")

print("second block started")
path = args.prompts
quantization = args.quantization
database_path = args.database
eval_results = args.results


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
import os
# Открываем промпты из папки llm-syntax/prompts по очереди

# Считываем системный промпт из файла
for filename in os.listdir(path):
    prompts = []
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            prompts.append(text)

eval_results = {"prompt": [], "temperature": [], "result": []}
print("second block ended")
    
import os

def chunks_from_conllu(database_path, chunk_size=100, output_prefix='chunk', output_dir='chunk_folder'):
    os.makedirs(output_dir, exist_ok=True)  # Создаёт папку, если её нет

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

    return chunk_files


# Сбор результатов

def collect_outputs(prompt, chunk_file):
    print(f"     Opening chunk file: {chunk_file}")
    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunk_text = f.read()
    print(f"     Chunk text length: {len(chunk_text)} characters")

    # Build the chat message format
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": chunk_text}],
            },
        ]
    ]
    print(f"     Messages created: {len(messages)} outer items, {len(messages[0])} inner messages")

    print("     Tokenizing messages...")
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception as e:
        print(f"     Tokenization failed: {e}")
        return

    input_keys = list(inputs.keys())
    print(f"     Input keys: {input_keys}")
    input_length = inputs['input_ids'].shape[1]
    print(f"     Input token length: {input_length}")

    temperature = random.uniform(0.4, 1.0)
    print(f"     Sampling temperature: {temperature:.2f}")

    print(f"     Model device: {model.device}")

    print("     Generating model output...")
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                temperature=temperature,
                top_k=50,
                max_new_tokens=32,
            )
        print(f"     Output tensor shape: {outputs.shape}")
    except Exception as e:
        print(f"     Generation failed: {e}")
        return

    print("     Decoding outputs...")
    try:
        decoded = tokenizer.batch_decode(outputs)
        print(f"     Decoded output length: {len(decoded[0])} characters")
    except Exception as e:
        print(f"     Decoding failed: {e}")
        return

    match = re.findall(r'<start_of_turn>(.*?)<end_of_turn>', decoded[0], re.DOTALL)
    result = match[1] if len(match) > 1 else "NO_MATCH"
    preview = result[:200] + ("..." if len(result) > 200 else "")
    print(f"     Parsed result: {preview}")

    eval_results['prompt'].append(prompt)
    eval_results['temperature'].append(temperature)
    eval_results['result'].append(result)
    print(f"     Result saved.\n")


chunk_files = chunks_from_conllu(database_path, chunk_size=100)

c=1
for prompt in prompts:
    for chunk_file in chunk_files:
        print(f"Processing chunk {c}/{len(chunk_files)}")
        collect_outputs(prompt, chunk_file)
        c += 1

df = pd.DataFrame(eval_results)
df.to_csv(eval_results, index=False)