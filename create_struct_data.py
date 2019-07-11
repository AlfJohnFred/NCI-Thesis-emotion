import os
import errno
import shutil


def segregate_data(base_dir, file_names_with_ext):
    data_male_normal_train = []
    data_female_normal_train = []
    data_male_high_train = []
    data_female_high_train = []
    data_male_normal_test = []
    data_female_normal_test = []
    data_male_high_test = []
    data_female_high_test = []

    final_array = []

    for file_names_with_ext in file_names_with_ext:
        file_name_without_ext = os.path.splitext(file_names_with_ext)[0]
        gender_bit = int(file_name_without_ext.split("-")[6])
        repetition_bit = file_name_without_ext.split("-")[5]
        emotional_intensity_bit = file_name_without_ext.split("-")[3]
        if (gender_bit % 2) == 0:
            if repetition_bit == '01':
                if emotional_intensity_bit == '01':
                    data_female_normal_train.append(base_dir + file_names_with_ext)
                else:
                    data_female_high_train.append(base_dir + file_names_with_ext)
            else:
                if emotional_intensity_bit == '01':
                    data_female_normal_test.append(base_dir + file_names_with_ext)
                else:
                    data_female_high_test.append(base_dir + file_names_with_ext)
        else:
            if repetition_bit == '01':
                if emotional_intensity_bit == '01':
                    data_male_normal_train.append(base_dir + file_names_with_ext)
                else:
                    data_male_high_train.append(base_dir + file_names_with_ext)
            else:
                if emotional_intensity_bit == '01':
                    data_male_normal_test.append(base_dir + file_names_with_ext)
                else:
                    data_male_high_test.append(base_dir + file_names_with_ext)

    final_array.append(data_male_normal_train)
    final_array.append(data_female_normal_train)
    final_array.append(data_male_high_train)
    final_array.append(data_female_high_train)
    final_array.append(data_male_normal_test)
    final_array.append(data_female_normal_test)
    final_array.append(data_male_high_test)
    final_array.append(data_female_high_test)

    if len(final_array) > 0:
        return final_array
    else:
        return -1


def create_data_dirs(final_base_dir, segregated_data):
    for index, item in enumerate(segregated_data):
        if index == 0:
            dir_name = "data_male_normal_train"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        elif index == 1:
            dir_name = "data_female_normal_train"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif index == 2:
            dir_name = "data_male_high_train"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif index == 3:
            dir_name = "data_female_high_train"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif index == 4:
            dir_name = "data_male_normal_test"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif index == 5:
            dir_name = "data_female_normal_test"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif index == 6:
            dir_name = "data_male_high_test"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif index == 7:
            dir_name = "data_female_high_test"
            try:
                os.makedirs(final_base_dir + dir_name)
                for file_name in segregated_data[index]:
                    src = file_name
                    dst = final_base_dir + dir_name + "\\" + file_name.split("\\")[6]
                    shutil.copy(src, dst)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise


if __name__ == '__main__':
    directory = "C:\\users\\dvada\\Desktop\\Dissertation\\Data\\"
    final_directory = "C:\\users\\dvada\\Desktop\\Dissertation\\FinalData\\"
    file_names = os.listdir(directory)
    segregated_files = segregate_data(directory, file_names)
    create_data_dirs(final_directory, segregated_files)
