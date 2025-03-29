import os

def check_format(file_path):
    _, extension = os.path.splitext(file_path)
    if extension.lower() == '.doc':
        return 'doc'
    elif extension.lower() == '.pdf':
        return 'pdf'
    elif extension.lower() == '.txt':
        return 'txt'
    else:
        return 'unknown'