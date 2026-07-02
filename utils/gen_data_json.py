import os, json, sys
from glob import glob

def main(data_dir):
    img_filelist = []
    for ext in ['.mhd','.nii.gz','.nii','.mha']:
        img_filelist += glob(os.path.join(data_dir,'*' + ext))
    #print(img_filelist)
    out_json = os.path.join(data_dir,'data.json')

    data_json = {'val':[]}

    for img_file in img_filelist:
        img_base = os.path.basename(img_file)
        data_json['val'].append({'image':img_base})

    json_object = json.dumps(data_json, indent=4)
 
    # Writing to sample.json
    with open(out_json, "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__':
    main(sys.argv[1])
