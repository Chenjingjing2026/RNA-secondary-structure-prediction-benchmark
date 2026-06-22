import numpy as np

class Evaluate:
    def __init__(self, predicted_structure_set, structure_set):
        self.predicted_structure_set = predicted_structure_set
        self.structure_set = structure_set

    def getscore(self):
        num_correct_pair = 0
        num_true_pair = 0
        num_predicted_pair = 0

        # Initialize TP, FP, TN, FN
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        print('len:',len(self.predicted_structure_set), len(self.structure_set))
        zips = zip(self.predicted_structure_set, self.structure_set)
        zipslist = list(zips)
        if not zipslist:
            print('none')
        else:
            print('mot not')
        for predicted_structure, true_structure in zip(self.predicted_structure_set, self.structure_set):
            #print(predicted_structure,true_structure)
            num_predicted_pair += predicted_structure.shape[0]
            print('true_structure',true_structure)
            num_true_pair += true_structure.shape[0]
            for predicted_pair in predicted_structure:
                for true_pair in true_structure:
                    if (predicted_pair == true_pair).all():
                        #print(predicted_pair,true_pair)
                        num_correct_pair+=1
        #print(num_correct_pair,num_predicted_pair,num_true_pair)
        Sensitivity = num_correct_pair/num_true_pair
        PPV = num_correct_pair/num_predicted_pair
        try:
            F_value = 2 * (Sensitivity * PPV) / (Sensitivity + PPV)
        except ZeroDivisionError:
            F_value = 0
            print("kotti??")
            print("ZeroDivisionError!!")

            # Calculate MCC
        TP = num_correct_pair  # True Positives
        FP = num_predicted_pair - num_correct_pair  # False Positives
        FN = num_true_pair - num_correct_pair  # False Negatives
        # TN is not directly available, but we can assume it as the total possible pairs minus TP, FP, and FN
        # This is a simplification and may not be accurate in all cases
        total_possible_pairs = num_true_pair * num_predicted_pair  # Example assumption
        TN = total_possible_pairs - (TP + FP + FN)  # True Negatives
        # 将整数转换为浮点数
        TP = float(TP)
        FP = float(FP)
        TN = float(TN)
        FN = float(FN)
        # MCC formula
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        if denominator != 0:
            MCC = (TP * TN - FP * FN) / denominator
        else:
            MCC = 0
        return Sensitivity,PPV,F_value, MCC
