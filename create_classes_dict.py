import json
with open('yns_spisok.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
lines = [line.strip() for line in lines]
filtered_lines = [line for line in lines if line.startswith('n') and ':' in line]
sorted_lines = sorted(filtered_lines, key=lambda x: int(x.split(':')[0][1:]))
classes_dict = {
    str(index): [line.split(':')[0], line.split(':')[1].strip()]
    for index, line in enumerate(sorted_lines)
}
with open('classes.json', 'w', encoding='utf-8') as json_file:
    json.dump(classes_dict, json_file, indent=4, ensure_ascii=False)
print(f"Словарь сохранён в файл 'classes.json'.")