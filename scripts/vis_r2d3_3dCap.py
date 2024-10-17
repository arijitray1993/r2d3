import pdb
import shutil
import sys
sys.path.append("../")
from custom_datasets.dataloaders import ProcTHOR_image_camposition_marked

if __name__ == "__main__":

    args = {
        'split': "val",
        'mode': "val",
        'include_children': False,
        'use_no_mark_baseline': True,
        'use_angle': True,
        'use_attributes': True,
        'normalize_rotation': True,
        'qa_format': True
    }

    dataset = ProcTHOR_image_camposition_marked(args, None, None)

    html_str = f"<html><head></head><body>"
    public_im_folder = "/net/cs-nfs/home/grad2/array/public_html/research/r2d3/3d_cap/"
    count= 0
    for entry in dataset:
        im_file, img, prompt, text_labels, program_text, obj_ids_present, dataname = entry
        im_file = im_file[0]
        # pdb.set_trace()
        html_str += f"<p>{text_labels}</p>"
        public_path_im = public_im_folder + "_".join(im_file.split("/")[-2:])
        shutil.copyfile(im_file, public_path_im)
        html_im_url = "https://cs-people.bu.edu/array/"+public_path_im.split("/net/cs-nfs/home/grad2/array/public_html/")[-1]
        html_str += f"<img src='{html_im_url}' style='width: 300px; height: 300px;'>"
        html_str += f"<p></p>"
        html_str += "<hr>"
        count += 1
        if count >= 50:
            break
    
    html_str += "</body></html>"
    with open("/net/cs-nfs/home/grad2/array/public_html/research/r2d3/3dCap.html", "w") as f:
        f.write(html_str)

