## Reproducibility

Much like how different tokenization schemes can result in slightly
different values for, e.g., BLEU, slightly different image
preprocessing can result in slightly different CLIPScore values. More
details are available
[here](https://github.com/jmhessel/clipscore/blob/main/README.md#reproducibility-notes). The
official version of CLIPScore is computed in float16 on GPU, and a
warning will be raised if you're using CPU (though the results should
be similar up to floating point precision). Another important
standarization is the particular image files --- because jpg
compression is lossy, small changes (e.g., different resizing, saving
jpgs multiple times, etc.) of images may result in slighly different
CLIPScores. This repo contains precomputed MSCOCO features extracted
on GPU from the 2017 train/val/test images (available
[here](https://cocodataset.org/#download)). For reproduability, you
can compare your MSCOCO images to [these
checksums](https://storage.googleapis.com/ai2-jack-public/clipscore/mscoco_checksum.txt.zip),
or just use the precomputed features on GPU.
