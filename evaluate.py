import argparse, re, random, os
import pandas as pd
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM


parser = argparse.ArgumentParser(description="Инструмент для парсинга синтаксической структуры")

parser.add_argument("--prompts", type=str, required=True, help="Путь к файлу с тестовыми данными")
parser.add_argument('--quantization', type=lambda x: x.lower() == 'true', default=False)

args = parser.parse_args()

path = args.prompts
quantization = args.quantization

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

# Подгрузка базы данных из файла
with open('llm-syntax/ru_syntagrus-ud-train-a.conllu') as f:
    database = f.read()

import os
# Открываем промпты из папки llm-syntax/prompts по очереди
path = args.prompts
directory = os.fsencode(path)

# Считываем системный промпт из файла
for filename in os.listdir(directory):
    prompts = []
    if filename.endswith(".txt"):
        with open(os.path.join(prompt_path, filename), "r", encoding="utf-8") as f:
            text = f.read()
            prompts.append(text)

eval_results = {"prompt": [], "temperature": [], "result": []}

def collect_outputs(prompt):
    # Формирование промпта: системная инструкция из файла склеивается с информацией из базы знаний и текущим промптом для оцеенки системы
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": database},]
            },
        ],
    ]
    # Формирование входных данных для модели
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs_storage = []

    for _ in range(len(database.split('\n'))):  # Количество итераций зависит от количества строк в базе данных
        eval_results['prompt'].append(prompt)
        # Рандомизировать температуру
        temperature = random.uniform(0.4, 1.0)
        eval_results['temperature'].append(temperature)
        # Инференс производится заданное пользователем количество раз итераций
        with torch.inference_mode():
            outputs = model.generate(**inputs, temperature=temperature, top_k=50, max_new_tokens=1024)

        outputs = tokenizer.batch_decode(outputs)
        
        # Добавляем парсинг ответа модели
        pattern = r'<start_of_turn>(.*?)<end_of_turn>'
        matches = re.findall(pattern, outputs[0], re.DOTALL)[1]
        eval_results['result'].append(matches)

for prompt in prompts:
    collect_outputs(prompt)
    
df = pd.DataFrame.from_dict(eval_results)
df.to_csv("evaluation/results.csv", index=False)
