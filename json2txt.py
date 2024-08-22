import os

def extract_image_paths(folder_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    f.write(os.path.join(root, file) + '\n')

if __name__ == "__main__":
    folder_path = input("请输入文件夹的路径：")
    output_file = input("请输入要保存的TXT文件的路径：")
    extract_image_paths(folder_path, output_file)