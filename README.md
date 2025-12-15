# Palette-based image Colorization 

Official implementation of [Exploring Palette based Color Guidance in Diffusion Models](https://arxiv.org/abs/2508.08754), ACM MM 2025.

![Overview_image](docs/overview.png)

## Introduction

This project presents a novel approach to image colorization using palette-based guidance within a diffusion model framework. Our method explores various palette representation techniques that are seamlessly integrated with text embeddings to control the image generation process.
The training phase of palette representation models are based on our previous work: [Multimodal Color 
recommendation in vector graphic documents](https://github.com/CyberAgentAILab/text2palette).

## Update
- [x] Inference code and paper are released.
- [ ] Palette embedding extraction and training code
- [ ] Training/validation/test datasets

## Setup

Dependencies
- GPU: NVIDIA A100-80G * 1

Install and requirements
```
conda env create -f environment.yaml
conda activate color-env
huggingface-cli login
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./clip-vit-large-patch14
conda install -y ipykernel
python -m ipykernel install --user --name color-env
```

### Quick demo

Download the pre-trained [model](https://storage.googleapis.com/ailab-public/palette2image/palette-image-model-plt-injected-epoch%3D29.ckpt) and place it in the `checkpoints/` directory
Run the demo notebook: `colorization.ipynb`
   - This notebook demonstrates colorized image generation with palette guidance from reference images
   - Uses pre-created palette embeddings for evaluation experiments

## Acknowledgements
This project is developped on the codebase of [ControlNet](https://github.com/lllyasviel/ControlNet). We appreciate their great work!

## Citation

```bibtex
@misc{qiu2025exploringpalettebasedcolor,
      title={Exploring Palette based Color Guidance in Diffusion Models}, 
      author={Qianru Qiu and Jiafeng Mao and Xueting Wang},
      year={2025},
      eprint={2508.08754},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2508.08754}, 
}
```
