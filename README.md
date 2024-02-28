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

## License

Our code is released under Apache License 2.0, unless stated otherwise.
