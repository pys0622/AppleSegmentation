import glob

label_files = glob.glob('dataset/labels/**/**/*.txt', recursive=True)

for file_path in label_files:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        tokens = line.strip().split()
        if tokens:
            tokens[0] = '0'  # 클래스 ID 1 → 0
            updated_lines.append(' '.join(tokens))

    with open(file_path, 'w') as f:
        f.write('\n'.join(updated_lines))