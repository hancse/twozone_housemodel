
# https://www.dangtrinh.com/2015/08/excel-to-list-of-dictionaries-in-python.html

import pandas as pd
import openpyxl
import string
# from ruamel import yaml
import yaml


def index_to_col(index):
    # return string.uppercase[index]
    return openpyxl.utils.cell.get_column_letter(index+1)


def excel_to_dict(excel_path, headers=[]):
    wb = openpyxl.load_workbook(excel_path)
    sheet = wb['Sheet1']
    result_dict = []
    for row in range(2, sheet.max_row+1):
        line = dict()
        for header in headers:
            cell_value = sheet[index_to_col(headers.index(header)) + str(row)].value
            if type(cell_value) == 'unicode':
                cell_value = cell_value.encode('utf-8').decode('ascii', 'ignore')
                cell_value = cell_value.strip()
            elif type(cell_value) is int:
                cell_value = str(cell_value)
            elif cell_value is None:
                cell_value = ''
            line[header] = cell_value
        result_dict.append(line)
    return result_dict


if __name__ == "__main__":
    # from excel_utils import excel_to_dict
    data = excel_to_dict('xl_for_yaml.xlsx', ['Element', 'Name', 'Capacity', 'Conductivity', 'Flow', 'NodeA', 'NodeB'])
    for d in data:
        print(d)
    with open("xl_for_yaml.yml", "w") as config_outfile:
        yaml.dump(data, config_outfile, indent=4, sort_keys=False)

    df = pd.read_excel('xl_for_yaml.xlsx')
    dd = df.T.to_dict()
    print(dd)
