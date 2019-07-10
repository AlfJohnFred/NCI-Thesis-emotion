import os


file_names_with_ext = os.listdir("C:\\users\\dvada\\Desktop\\Dissertation\\Data\\Actor_01")
counter = 1
file_names = []
for file_names_with_ext in file_names_with_ext:
    file_names.append(os.path.splitext(file_names_with_ext)[0])

for file_names in file_names:
    print(file_names.split("-")[5:7])