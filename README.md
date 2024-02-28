# UNMHG
EACL 2024 (findings) paper: "Towards Unified Uni- and Multi-modal News Headline Generation"


![UNMHG-overview](./resources/unmhg_overview.png?raw=true)


# Code

We include (some) of the code used in our experiments. It is structured as follows:

```
├── model
│   ├── modeling_umms_blip2.py - BLIP-2 extended to handle articles with multiple images and text-only articles
│   └── model_BLIP2_umms.py - Trainer class for the T5CLIP models
│   └── model_umms.py - Trainer class for the modified BLIP-2 model
|── preprocess
|   ├── filter_and_split_PENS.py - pre-processing of the PENS dataset


```

# Data

The text-only `PENS` dataset can be obtained [here](https://msnews.github.io/pens_data.html), the video-based `MLASK` [here](http://hdl.handle.net/11234/1-5135) and the images-based `M3LS` [here](https://github.com/Raghvendra-14/M3LS).


In the paper, in Appendix A.3, we describe the procedure of collecting the image targets for M3LS articles. The textual data (articles/titles/abstracts) and source images should be collected from the [original repository](https://github.com/Raghvendra-14/M3LS). In `data/M3LS` we are sharing a `M3LS_ref_images.tsv` file with two columns: **HASH** (a hashed ID that can be used to match with the corresponding article in M3LS) and **REF_IMAGE_URL** (points to the image target, i.e., the pictorial summary). Once you collect the images (using e.g., `wget`), you should process them with `data/M3LS/crop_reference_images.py`. This script removes a horizontal strip from the bottom of the image. The reason is a watermark (open one of the images, and see for yourself). We remove it, as it would give the model an easy clue for distuingishing the target images.


## License

Our code is released under Apache License 2.0, unless stated otherwise.
