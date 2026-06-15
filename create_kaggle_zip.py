import os
import zipfile

def create_kaggle_zip():
    items_to_zip = ['app', 'experiments', 'data', 'requirements.txt']
    zip_filename = 'kaggle_upload_v3.zip'
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in items_to_zip:
            if not os.path.exists(item):
                continue
            if os.path.isfile(item):
                arcname = item.replace(os.sep, '/')
                zipf.write(item, arcname)
            else:
                for root, _, files in os.walk(item):
                    for file in files:
                        if file.endswith('.pyc') or '__pycache__' in root:
                            continue
                        file_path = os.path.join(root, file)
                        # Force forward slashes for Kaggle
                        arcname = file_path.replace(os.sep, '/')
                        zipf.write(file_path, arcname)
    print(f"Successfully created {zip_filename} with fixed scripts and forward slashes!")

if __name__ == "__main__":
    create_kaggle_zip()
