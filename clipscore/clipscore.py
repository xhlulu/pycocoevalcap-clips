from __future__ import division
import os
import sys
import subprocess
import threading
import json
import numpy as np
import ast
import tempfile
import clip
import torch
from zipfile import ZipFile
from urllib.request import urlretrieve

# The cache dir is where we will store all of the temporary
# data for CLIP
CLIPDIR = os.path.dirname(__file__)

def print_progress(transferred_blocks, block_size, total_size):
    current_mb = transferred_blocks * block_size / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    percent = current_mb / total_mb
    progress_str = "Progress: {:5.1f}M / {:5.1f}M ({:6.1%})"
    print(progress_str.format(current_mb, total_mb, percent), end='\r')

    
class ClipScore:
    """
    Main Class to compute CLIPScore
    pip install git+https://github.com/openai/CLIP.git
    """

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print('clipscore is using {}'.format(device))
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        self.model = model
        cwd = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(cwd, CLIPDIR)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.mscoco_feats_path = os.path.join(CLIPDIR, 'all_coco_captions_caption_features~ViT-B32.npy')
        self.mscoco_id2row_path = os.path.join(CLIPDIR, 'all_coco_captions_caption_features~ViT-B32~cap2row.json')
        if not os.path.exists(self.mscoco_feats_path):
            url = 'https://storage.googleapis.com/ai2-jack-public/clipscore/mscoco_vitb_features.zip'
            zip_file, headers = urlretrieve(url, reporthook=print_progress)
            for filef in ['all_coco_captions_caption_features~ViT-B32.npy', 'all_coco_captions_caption_features~ViT-B32~cap2row.json']:
                ZipFile(zip_file).extract(filef, CLIPDIR)
        

    def compute_score(self, gts, res):
        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())
        
        input_data = []
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            input_data.append({
              "image_id" : id,
              "test" : hypo[0],
              "refs" : ref
            })
        

        cwd = os.path.dirname(os.path.abspath(__file__))
        temp_dir=os.path.join(cwd, TEMP_DIR)
        if not os.path.exists(temp_dir):
          os.makedirs(temp_dir)
        in_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir,
                                              mode='w+')
        json.dump(input_data, in_file, indent=2)
        in_file.close()

        # Start job
        out_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        out_file.close()
        cache_dir=os.path.join(cwd, CACHE_DIR)
        if not os.path.exists(cache_dir):
          os.makedirs(cache_dir)
        spice_cmd = ['java', '-jar', '-Xmx8G', SPICE_JAR, in_file.name,
          '-cache', cache_dir,
          '-out', out_file.name,
          '-subset',
          '-silent'
        ]
        subprocess.check_call(spice_cmd, 
            cwd=os.path.dirname(os.path.abspath(__file__)))

        # Read and process results
        with open(out_file.name) as data_file:    
          results = json.load(data_file)
        os.remove(in_file.name)
        os.remove(out_file.name)

        imgId_to_scores = {}
        spice_scores = []
        for item in results:
          imgId_to_scores[item['image_id']] = item['scores']
          spice_scores.append(self.float_convert(item['scores']['All']['f']))
        average_score = np.mean(np.array(spice_scores))
        scores = []
        for image_id in imgIds:
          # Convert none to NaN before saving scores over subcategories
          score_set = {}
          for category,score_tuple in imgId_to_scores[image_id].items():
            score_set[category] = {k: self.float_convert(v) for k, v in score_tuple.items()}
          scores.append(score_set)
        return average_score, scores

    def method(self):
        return "SPICE"

def tmp_main():
    with open('cached_gts.json') as f:
        cached_gts = json.load(f)
    with open('cached_res.json') as f:
        cached_res = json.load(f)

    cs = ClipScore()
    cs.compute_score(cached_gts, cached_res)
    

if __name__ == '__main__':
    tmp_main()
