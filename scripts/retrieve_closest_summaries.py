import json
from sentence_transformers import SentenceTransformer

from numpy import dot
from numpy.linalg import norm
import tqdm

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

if __name__=="__main__":

    summary_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/all_room_json_programs_ai2_train_room_summaries_gtobjonly.json", "r"))

    caption_data = json.load(open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/all_room_json_programs_ai2_train_room_captions_gtobjonly.json", "r"))
    model = SentenceTransformer('sentence-transformers/sentence-t5-base')

    caption2embed = {}
    for program_text, house_json, caption in summary_data:
        embeddings = model.encode(caption)
        caption2embed[caption] = embeddings

    closest_summary_data = []
    for ind, (program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions) in enumerate(tqdm.tqdm(caption_data)):
        
        summary = summary_data[ind][2]

        summary_embedding = caption2embed[summary]

        #get top 5 closest summary embeddings
        closest_summaries = []
        for other_inds in range(len(summary_data)):
            if other_inds == ind:
                continue
            other_summary = summary_data[other_inds][2]
            caption_embedding = caption2embed[other_summary]
            similarity = float(cosine_similarity(caption_embedding, summary_embedding))
            closest_summaries.append((other_inds, similarity))

        closest_summaries.sort(key=lambda x: x[1], reverse=True)

        closest_summaries = closest_summaries[:5]

        closest_summary_data.append((program_text, house_json, cam_ind_to_position, all_imgs, all_objs, all_room_captions, summary, closest_summaries))

    json.dump(closest_summary_data, open("/projectnb/ivc-ml/array/research/robotics/dreamworlds/custom_datasets/procThor/all_room_json_programs_ai2_train_room_closest_summaries_gtobjonly.json", "w"))