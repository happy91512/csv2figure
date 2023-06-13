import os
import pandas as pd
from matplotlib import pyplot as plt
from Useful_Tools_for_Me.FileTools.FileOperator import check2create_dir

def find_folder_file(dir_path:str, path_length:int):
        target_path = []
        for file in os.listdir(dir_path):
            if len(file) > path_length:
                target_path.append(os.path.join(dir_path, file))
        return target_path

def get_save_info(path:str):
    #{train / val}_{loss / acc}_{label}.png
    #task, number, target_label
    split = path.split('/')
    task, number = split[-1][:-4].split('_')
    return task, number

def plot_single_model(all_csv_path:list[str], save_folder:str):
    plt.figure(figsize=(8, 6))
    for csv_path in all_csv_path:
        task, number = get_save_info(csv_path)
        df = pd.read_csv(csv_path)
        columns = df.columns[1:]
        for target_label in columns:
            y = df[target_label]
            plt.plot(df.index, y)
            plt.title(f"{target_label}") # title
            plt.xlabel('epoch') # x label
            plt.ylabel(f"{number}") # y label
            plt.savefig(save_folder + f"/{task}_{number}_{target_label}.png")
            plt.clf()

def plot_multi_model(all_csv_path:list[str], save_folder:str):
    columns = pd.read_csv(all_csv_path[0]).columns[1:]
    plt.figure(figsize=(8, 6))
    for target_label in columns:
        x_axis_lim = 10000
        for i, csv_path in enumerate(all_csv_path):
            # print(csv_path)
            opt = csv_path.split('_')[-3][:-8]
            lr = csv_path.split('_')[-3][-8:]
            s = csv_path.split('_')[-2][4]
            try:
                i = int(csv_path.split('/')[-2][-1])
                label = f"{opt}-{lr}_S{s}"
            except:
                label = f"{opt}-{lr}_S{s}_handcraft"
            model_name = csv_path.split('/')[-2]
            task, number = get_save_info(csv_path)
            df = pd.read_csv(csv_path)
            if len(df) < x_axis_lim:
                x_axis_lim = len(df)
            y = df[target_label]
           
            plt.plot(df.index, y, label = label)
        if len(all_csv_path) >= 6:
            y_position = 1.2
        else:
            y_position = 1.1 + len(all_csv_path)/60
        plt.legend(bbox_to_anchor=(1.0, y_position), fontsize = '8', borderaxespad = 0.5)
        plt.title(f"{target_label}") # title
        plt.xlim(0, x_axis_lim - 1)
        plt.xlabel('epoch') # x label
        plt.ylabel(f"{number}") # y label
        plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
        # if csv_path.split('/')[-2][-5:] == 'craft':
        #     print(True)
        #     plt.savefig(save_folder + f"/{task}_{number}_{target_label}_Handcraft.png")
        # if csv_path.split('/')[-2][-5:] != 'craft':
        #     print(False)
        plt.savefig(save_folder + f"/{task}_{number}_{target_label}.png")
        plt.cla()

if __name__ == "__main__":
    check2create_dir('./plot_output/single_model/')
    check2create_dir('./plot_output/all_model/')

    folder_list = find_folder_file(dir_path = './src_csv/', path_length = 30)
    multi_targets = [[], [], [], []]
    name_list = ["/train_acc.csv", "/val_acc.csv", "/train_loss.csv", "/val_loss.csv"]

    for folder in folder_list:#Plotting the curve diagram for each label of every model
        model_name = folder.split('/')[-1]
        single_model_save_folder = 'plot_output/single_model/' + model_name
        multi_model_save_folder = 'plot_output/all_model/' 
        train_val_compare_savve_folder = 'plot_output/train_val_compare/'+ model_name
        check2create_dir(single_model_save_folder)
        check2create_dir(multi_model_save_folder)
        save_target = [folder + name for name in name_list]
        plot_single_model(save_target, single_model_save_folder)  
        for i in range(4):
            multi_targets[i].append(folder + name_list[i])
    for targets in multi_targets:         
        plot_multi_model(targets, multi_model_save_folder)
 