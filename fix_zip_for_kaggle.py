import zipfile

input_zip = 'kaggle_upload.zip'
output_zip = 'kaggle_upload_fixed.zip'

with zipfile.ZipFile(input_zip, 'r') as zin:
    with zipfile.ZipFile(output_zip, 'w') as zout:
        for item in zin.infolist():
            buffer = zin.read(item.filename)
            item.filename = item.filename.replace('\\', '/')
            zout.writestr(item, buffer)
print(f"Successfully created fixed zip file: {output_zip}")
