
import numpy as np
import csv


def house_to_csv():
    with open('new_yaml_xl/simulation_output.csv', mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=';',
                                   quotechar='"', quoting=csv.QUOTE_MINIMAL)

        output_writer.writerow(['Description',
                                'Resultaten HAN Dynamic Model Heat Built Environment'])
        output_writer.writerow(['Chain number', '1'])
        output_writer.writerow(['Designation', '2R-2C-1-zone' '2R-2C-1-zone'])
        output_writer.writerow(['Node number', '0', '1', '2'])
        output_writer.writerow(['Designation', 'Outside Temperature',
                               'Internals', 'Load bearing construction'])
        output_writer.writerow(['TIMESTEP', 'TEMPERATURE (GRADEN C)', 'TEMPERATURE (GRADEN C)',
        'SOLAR (J)', 'SOURCES (J)', 'HEATING (J)', 'TEMPERATURE', 'SOLAR', 'SOURCES', 'HEATING'])


if __name__ == "__main__":
    pass
    # house_to_csv()


