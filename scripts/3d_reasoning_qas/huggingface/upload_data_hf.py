import json
import pandas as pd
from huggingface_hub import HfApi, HfFolder, Repository

def get_qa_type(question):
    question_type = "other"
    
    if "how did the camera" in question.lower() or "is the camera moving" in question.lower():
        question_type = "action_sequence"

    if ("need to go" in question.lower()):
        question_type = "goal_aim"

    if "any of the objects in the initial" in question.lower():
        question_type = "obj_movement"

    if "if i" in question.lower():
        question_type = "action_consequence"

    if 'if i move to the' in question.lower() or "for someone at the" in question.lower():
        question_type = "perspective"

    
    return question_type


if __name__=="__main__":

    split = "val"

    if split=="train":
        spatial_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_spatial_qas_v2_train.json' 
        spatial_data = json.load(open(spatial_qa_json_path))   

        complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2.json' # remove v2 for prev version.
        complex_qa_json_path_split2 = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_train_v2_split2.json'
        complex_data = json.load(open(complex_qa_json_path)) + json.load(open(complex_qa_json_path_split2))
        #complex_data = random.sample(complex_data, 6900)

        camera_move_path = "/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_cameramove_qas_train.json"
        camera_move_data = json.load(open(camera_move_path))

        perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
        perspective_data = json.load(open(perspective_qa_json_path))
        #perspective_data = random.sample(perspective_data, 300) 

        print("num images in spatial data:", len(spatial_data))
        print("num images in complex data:", len(complex_data))
        print("num images in perspective data:", len(perspective_data))
        all_complex_data = []

        for house_ind, cam_pos, cam_rot, qa_entries in spatial_data:
            for question, im_order, answers in qa_entries:
                all_complex_data.append((question, im_order, answers))

        for house_ind, cam_pos, cam_rot, qa_entries in complex_data:
            for question, im_order, answers in qa_entries:
                question = question.replace("turn look straight", "look straight")

                if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                    new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                    answers = new_answers
                
                if "how did the camera likely move" in question.lower():
                    question = question.replace("How did the camera likely move when shooting the video", "How did the camera rotate from the first image to the second image") 

                all_complex_data.append((question, im_order, answers))
        
        for _,_,_, qa_entries in camera_move_data:        
            self.data.extend(qa_entries)

        
        perspective_count = 0
        qa_len_hist = defaultdict(int)
        for _,_,_, qa_entries in perspective_data:
            qa_len_hist[len(qa_entries)] += 1
            for question, im_order, answers in qa_entries:
                if random.random() > 0.05:
                    continue
                question = question.replace("turned towards the", "facing 90 degrees to the")
                question = question.replace("turned right", "turned right by 90 degrees")
                question = question.replace("turned left", "turned left by 90 degrees")
                perspective_count+=1
                all_complex_data.append((question, im_order, answers))
        print("pers count", perspective_count)
        print(qa_len_hist)
        
    else:
        complex_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/3d_navigation_qas_val_v2.json' # remove v2 for prev version.
        complex_data = json.load(open(complex_qa_json_path))

        perspective_qa_json_path = '/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/perspective_qas.json'
        perspective_data = json.load(open(perspective_qa_json_path))

        print("num images in complex data:", len(complex_data))
        
        all_complex_data = []
        all_action_consequence_data = []
        for house_ind, cam_pos, cam_rot, qa_entries in complex_data[int(len(complex_data)*0.1):]:
            for question, im_order, answers in qa_entries:
                question = question.replace("turn look straight", "look straight")

                if answers[0] == "rotated left and rotated right" or answers[0] == "rotated right and rotated left": # bug fix
                    new_answers = ["did not move", random.choice(["rotated left", "rotated right"])]
                    answers = new_answers
                
                qa_type = get_qa_type(question)
                #if qa_type == "action_consequence":
                #    all_action_consequence_data.append((question, im_order, answers))
                #else: 
                all_complex_data.append((question, im_order, answers))
        
        #all_complex_data += random.sample(all_action_consequence_data, 1000)

        perspective_count = 0
        pers_im_count = 0
        for _,_,_, qa_entries in perspective_data[int(len(perspective_data)*0.1):]:
            pers_im_count += 1
            for question, im_order, answers in qa_entries:
                question = question.replace("turned towards the", "facing 90 degrees to the")
                question = question.replace("turned right", "turned right by 90 degrees")
                question = question.replace("turned left", "turned left by 90 degrees")

                all_complex_data.append((question, im_order, answers))
                perspective_count += 1
                if perspective_count >= 779:
                    break
            if perspective_count >= 779:
                break

        print("num images in perspective data:", pers_im_count)

    pdb.set_trace()

    image_bytes_list = []
    questions_list = []
    answers_list = []
    question_types_list = []
    correct_answer_list = []
    # pdb.set_trace()
    for question, image_paths, answer_choices in all_complex_data:
        image_bytes = []
        for image_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                image_bytes.append(img_bytes)
        image_bytes_list.append(image_bytes)
        questions_list.append(question)
        answers_list.append(answer_choices)
        question_types_list.append(get_qa_type(question))
        correct_answer_list.append(answer_choices[0])
    
    df = pd.DataFrame({"image_bytes": image_bytes_list, "question": questions_list, "answers": answers_list, "question_type": question_types_list, "correct_answer": correct_answer_list})
    df.to_parquet(f"SAT_{split}.parquet")

    pdb.set_trace()

    api = HfApi()
    token = HfFolder.get_token()  # or set HF_TOKEN environment variable

    repo_id = "array/SAT"  # Change 'username' to your HF username

    # Create the dataset repository on the Hub (if it doesn't exist)
    # api.create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)

    # Clone the newly created dataset repo locally
    repo = Repository(local_dir=".", clone_from=repo_id)

    # Move the Parquet file into the repository directory
    import shutil
    shutil.move("my_image_dataset.parquet", "my_image_dataset/my_image_dataset.parquet")

    # Create a simple README for the dataset
    with open("my_image_dataset/README.md", "w") as f:
        f.write("# My Image Dataset\n\nThis dataset contains images stored in a Parquet file.\n")

    # Commit and push changes
    repo.git_add(all=True)
    repo.git_commit("Initial commit of the Parquet-based image dataset")
    repo.git_push()
    