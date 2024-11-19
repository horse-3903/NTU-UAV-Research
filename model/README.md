---
license: mit
tags:
- vision
pipeline_tag: depth-estimation
---

# ZoeDepth (fine-tuned on NYU and KITTI) 

ZoeDepth model fine-tuned on the NYU and KITTI datasets. It was introduced in the paper [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288) by Shariq et al. and first released in [this repository](https://github.com/isl-org/ZoeDepth).

ZoeDepth extends the [DPT](https://huggingface.co/docs/transformers/en/model_doc/dpt) framework for metric (also called absolute) depth estimation, obtaining state-of-the-art results.

Disclaimer: The team releasing ZoeDepth did not write a model card for this model so this model card has been written by the Hugging Face team.

## Model description

ZoeDepth adapts [DPT](https://huggingface.co/docs/transformers/en/model_doc/dpt), a model for relative depth estimation, for so-called metric (also called absolute) depth estimation.

This means that the model is able to estimate depth in actual metric values.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/zoedepth_architecture_bis.png"
alt="drawing" width="600"/>

<small> ZoeDepth architecture. Taken from the <a href="https://arxiv.org/abs/2302.12288">original paper.</a> </small>

## Intended uses & limitations

You can use the raw model for tasks like zero-shot monocular depth estimation. See the [model hub](https://huggingface.co/models?search=Intel/zoedepth) to look for
other versions on a task that interests you.

### How to use

The easiest is to leverage the pipeline API which abstracts away the complexity for the user:

```python
from transformers import pipeline
from PIL import Image
import requests

# load pipe
depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")

# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# inference
outputs = depth_estimator(image)
depth = outputs.depth
```
For more code examples, we refer to the [documentation](https://huggingface.co/transformers/main/model_doc/zoedepth.html#).

### BibTeX entry and citation info

```bibtex
@misc{bhat2023zoedepth,
      title={ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth}, 
      author={Shariq Farooq Bhat and Reiner Birkl and Diana Wofk and Peter Wonka and Matthias Müller},
      year={2023},
      eprint={2302.12288},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```