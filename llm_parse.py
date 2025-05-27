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
output_path = args.output

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
with open('prompts/short_glossary.txt') as f1:
    system_prompt_1 = f1.read()
# Задача: подставить сюда еще 2 промпта под новыми переменными
## TODO ##
with open('prompts/long_glossary_prompt.txt') as f2:
    system_prompt_2 = f2.read()

with open('prompts/example_after_glossary_prompt.txt') as f3:
    system_prompt_3 = f3.read()
# Системные роли удобнее подгружать из отдельного файла
# Задача: собрать это всё в функцию parse(system_role, user_prompt), кот. принимает на вход system prompt и выдает matches[1]
## TODO ##
def parse(system_prompt, user_prompt):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        },
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
    return matches[1]
 # Заменить на return
## Здесь функция будет заканчиваться ##

# Дописать код: применить созданную функцию parse() трижды 

def main():
    # Здесь мы считываем информацию из файла, указанного в input
    with open(input_path, 'r') as f:
        user_prompt = f.read()
      # Затем мы применяем функцию parse() трижды, чередуя системные промпты
    result1 = parse(system_prompt_1, user_prompt)
    result2 = parse(system_prompt_2, user_prompt)
    result3 = parse(system_prompt_3, user_prompt)
  
    # Далее записываем выдачу с каждой функции в файл, указанный в output 
    with open(output_path, 'w'):
        f.write("Вариант разметки 1\n")
        f.write(result1 + "\n\n")
        f.write("Вариант разметки 2\n")
        f.write(result2 + "\n\n")
        f.write("Вариант разметки 3\n")
        f.write(result3 + "\n")
    
if __name__ == "__main__":
    main()
