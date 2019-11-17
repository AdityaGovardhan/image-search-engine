import pprint

from data_preprocessing import DataPreProcessor
from pyfiglet import Figlet
from tabulate import tabulate
from termcolor import colored
from task_manager import TaskManager

f = Figlet()
print(colored(f.renderText('MWDB'), "red"))
task_3_top_k = None
pp = pprint.PrettyPrinter(indent=4)
print(colored('*************************Welcome to CSE 515: MWDB Project phase 2*************************', 'blue'))
while 1:
    print(colored(f.renderText('Phase - 2'), "blue"))
    print(tabulate([["1)", "Start Data Preprocessing"], ["2)", "Go For Tasks"], ["3)", "Exit"]],
                   headers=['Number', "Options"],
                   tablefmt='orgtbl'))
    print(colored("Select the option", "blue"))
    option = int(input())
    if option == 1:
        print(colored("Data Preprocessing Started", "blue"))
        data_preprocessor = DataPreProcessor()
        print(colored("Data Preprocessing Completed", "blue"))

    elif option == 2:
        while 1:
            feature_model_dict = {1: "color_moments",
                                  2: "local_binary_pattern",
                                  3: "histogram_of_gradients",
                                  4: "scale_invariant_feature_transformation"}
            dimentionality_reduction_tech_dict = {1: "pca", 2: "svd", 3: "nmf", 4: "lda"}
            print(colored("Select the which task you want to perform", "blue"))
            print(tabulate([['1)', 'Task 1'],
                            ['2)', 'Task 2'],
                            ['3)', 'Task 3'],
                            ['4)', 'Task 4'],
                            ['5)', 'Task 5'],
                            ['6)', 'Task 6'],
                            ['7)', 'Task 7'],
                            ['8)', 'Task 8'],
                            ['9)', 'Exit']],
                           headers=['Number', "Options"],
                           tablefmt='orgtbl'))
            task = int(input())
            if task in range(0, 9):
                print(colored(f.renderText('Task {0}'.format(task)), "blue"))
            if task in range(0, 6):
                print(tabulate([["1)", "CM"],
                                ["2)", "LBP"],
                                ["3)", "HOG"],
                                ["4)", "SIFT"]], headers=['Number', "Algorithm"],
                               tablefmt='orgtbl'))
                print(colored("Enter which Feature model you want to use", "blue"))
                feature_model_input = int(input())
                feature_model = feature_model_dict[feature_model_input]
                print(tabulate([["1)", "principal component analysis (PCA)"],
                                ["2)", "singular value decomposition (SVD)"],
                                ["3)", "non-negativematrix factorization (NMF)"],
                                ["4)", "latent dirichlet analysis (LDA)"]], headers=['Number', "Technique"],
                               tablefmt='orgtbl'))
                print(colored("Enter which Technique you want to use", "blue"))
                technique = int(input())
                dimentionality_reduction_tech = dimentionality_reduction_tech_dict[technique]
                print("Enter how many top latent semantics you want")
                top_k = int(input())
            if task in range(3, 6):
                label_dict = {
                              1: 'left',
                              2: 'right',
                              3: 'dorsal',
                              4: 'palmar',
                              5: 1,
                              6: 0,
                              7: 'male',
                              8: 'female'
                            }
                print("Please enter name of the label")
                print(tabulate([["1)", "left-hand"],
                                ["2)", "right-hand"],
                                ["3)", "dorsal"],
                                ["4)", "palmar"],
                                ["5)", "with accessories"],
                                ["6)", "without accessories"],
                                ["7)", "male"],
                                ["8)", "female"]], headers=['Number', "Label"],
                               tablefmt='orgtbl'))
                label_input = int(input())
                if label_input in range(1, 5):
                    label_type = 'aspect'
                elif label_input in range(5, 7):
                    label_type = 'accessories'
                elif label_input in range(7, 9):
                    label_type = 'gender'
                label = label_dict[label_input]

            task_manager = TaskManager()
            if task == 1:
                task_manager.execute_task1(feature_model, dimentionality_reduction_tech, top_k)
            elif task == 2:
                print("Please enter image id")
                image_id = str(input())
                print("Please enter number of similar images required")
                top_m = int(input())
                task_manager.execute_task2(feature_model, dimentionality_reduction_tech, top_k, top_m, image_id)
            elif task == 3:
                task_3_top_k = top_k
                task_manager.execute_task3(feature_model, dimentionality_reduction_tech, top_k, label, label_type)
            elif task == 4:
                print("Please enter image id")
                image_id = input()
                print("Please enter number of similar images required")
                top_m = int(input())
                task_manager.execute_task4(feature_model, dimentionality_reduction_tech, top_k, top_m, image_id, label, label_type)
            elif task == 5:
                print("Please enter unlabelled image id")
                image_id = input()
                task_manager.execute_task5(feature_model, dimentionality_reduction_tech, top_k, image_id, label,
                                           label_type, "CC_with_mean_radius")
            elif task == 6:
                print("Please enter subject id:")
                subject_id = int(input())
                print("Please Choose one approach")
                print(tabulate([["1)", "Mean/Aggregation"],
                                ["2)", "Weighted Average based on Category"]], headers=['Number', "Approach"],
                               tablefmt='orgtbl'))
                approach = int(input())
                task_manager.execute_task6(subject_id, approach)
            elif task == 7:
                print("Please enter how many similar subjects you want to find")
                k_top_sub_ids = int(input())
                print("Please Choose one approach")
                print(tabulate([["1)", "Mean/Aggregation"],
                                ["2)", "Weighted Average based on Category"]], headers=['Number', "Approach"],
                               tablefmt='orgtbl'))
                approach = int(input())
                task_manager.execute_task7(k_top_sub_ids, approach)
            elif task == 8:
                print("Please enter K value")
                k_top_sub_ids = int(input())
                task_manager.execute_task8(k_top_sub_ids)
            elif task == 9:
                print(colored(f.renderText('Terminating code...!!'.format(task)), "red"))
                break
            else:
                print("Please enter valid task number")
    elif option == 3:
        break
