import os

path = os.getcwd()

to_import = []
for tup in os.walk('../skxray'):
    if 'test' in tup[0]:
        continue
    print(tup)
    if '__init__.py' in tup[2]:
        package = '.'.join(tup[0].split(os.sep)[1:])
        to_import.append(package)
        for mod in tup[2]:
            if mod == '__init__.py':
                continue
            to_import.append(package + '.' + mod.split('.')[0])

print('----- Add this to the conda-recipe -----')
for imp in to_import:
    print('    - %s' % imp)
