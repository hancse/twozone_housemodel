# https://pythonexamples.org/python-replace-multiple-spaces-with-single-space-in-text-file/

import re


def remove_spaces(filename: str):
    with open(filename, "r+") as file_obj:
        data = file_obj.readlines()
        file_obj.seek(0)
        ms = re.compile('  ')  # two spaces ok, 4 spaces too much
        for line in data:
            line = re.sub(ms, ' ', line)
            file_obj.writelines(line)
        file_obj.truncate()

    # Check if file is closed
    if not file_obj.closed:
        print('File is not closed')
    else:
        print('File is closed')

if __name__ == "__main__":
    remove_spaces("../../excel_for_companies.yaml")


