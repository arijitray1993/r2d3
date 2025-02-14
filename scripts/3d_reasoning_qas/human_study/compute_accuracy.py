from collections import defaultdict
import csv
import numpy as np
import pdb
import statistics 

if __name__=="__main__":

    csv_file = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/scripts/3d_reasoning_qas/human_study/results/Batch_411206_batch_results_more.csv'
    
    # read the csv as dictionary
    with open(csv_file, mode='r') as infile:
        reader = csv.DictReader(infile)
        data = list(reader)

    accuracy = defaultdict(list)
    all_accuracy = []
    for item in data:
        # these are the keys: Input.Question	Input.Answer1	Input.Answer2	Input.Correct	Answer.Choice1	Answer.Choice2
        hitid = item['HITId']
        question = item['Input.question']
        answer1 = item['Input.answer1']
        answer2 = item['Input.answer2']
        correct = item['Input.correct']
        choice1 = item['Answer.answer.choice1']
        choice2 = item['Answer.answer.choice2']

        # pdb.set_trace()
        if correct == "1":
            if choice1 == "true":
                accuracy[hitid].append(1)
                all_accuracy.append(1)
            else:
                accuracy[hitid].append(0)
                all_accuracy.append(0)
        else:
            if choice2 == "true":
                accuracy[hitid].append(1)
                all_accuracy.append(1)
            else:
                accuracy[hitid].append(0)
                all_accuracy.append(0)
    
    print("All accuracy: ", np.mean(all_accuracy))
    
    final_accuracy = []
    # get majority vote
    for key in accuracy.keys():
        
        #if len(accuracy[key]) > 2:
            # pdb.set_trace()
        final_accuracy.append(statistics.mode(accuracy[key]))

    print("Accuracy: ", np.mean(final_accuracy))
    