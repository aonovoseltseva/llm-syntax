import os, argparse, re
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
import torch

parser = argparse.ArgumentParser(description="LLM Syntax Parser")
parser.add_argument("--eval", type=bool, required=True, help="Evaluaton mode (T/F)")
parser.add_argument("--input", type=str, required=True, help="Filepath to the input data")
parser.add_argument("--output", type=str, required=True, help="Output path")
args = parser.parse_args()
eval_mode = args.eval
input_path = args.input
output_parg = args.output

model_id = "google/gemma-3-1b-it"

# Применяем квантизацию: загружаем модель меньшей размерности
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Инициализация модели из HuggingFace: загружается локально на наше устройство
# Это значит, что она не использует сторонние сервисы, а все вычисления выполняются у нас
model = Gemma3ForCausalLM.from_pretrained(
    model_id, quantization_config=quantization_config
).eval()

# Токенизация тоже производится локально, т.е. на нашем устройстве
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Подгрузка промптов с файла
with open('prompts/short_glossary.txt') as f:
    system_prompt = f.read()
# Задача: подставить сюда еще 2 промпта под новыми переменными
## TODO ##

# Системные роли удобнее подгружать из отдельного файла
# Задача: собрать это всё в функцию parse(system_role, user_prompt), кот. принимает на вход system prompt и выдает matches[1]
## TODO ##
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": input_path},]
        },
    ],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=1024)

outputs = tokenizer.batch_decode(outputs)

# Добавляем парсинг ответа модели
pattern = r'<start_of_turn>(.*?)<end_of_turn>'
matches = re.findall(pattern, outputs[0], re.DOTALL)

print(matches[1]) # Заменить на return
## Здесь функция будет заканчиваться ##

# Дописать код: применить созданную функцию parse() трижды 

def main():
    # Здесь мы считываем информацию из файла, указанного в input
    # Затем мы применяем функцию parse() трижды, чередуя системные промпты
    # Далее записываем выдачу с каждой функции в файл, указанный в output 
    pass

if __name__ == "__main__":
    main()
