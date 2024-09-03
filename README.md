# AdvancedFastDataAcquisition-ObjDetSeg
Additional materials for our ISCV24 paper "Advanced Post-Processing for Object Detection Dataset Generation".


## Introduction
Here are additional resources to reproduce our results. Most importantly, we provide our configuration files for LoRA training and our LoRAs, trained to reproduce the YCB-V household objects with Stable Diffusion 1.5. In case you want to train your own LoRAs, we provide our configuration files as used with https://github.com/bmaltais/kohya_ss. 

## Usage
1. Get the luminance key version of YCB-V (YCB-V Luma) at https://huggingface.co/datasets/tpoellabauer/YCB-V-LUMA/tree/main. You can use code from https://github.com/tpoellabauer/FastDataAcquisition-ObjDetSeg to extract images and masks from the recordings. 
2. Setup https://github.com/AUTOMATIC1111/stable-diffusion-webui.
3. Put the provided LoRAs to your installation folder.
4. Run webui with your favourite SD1.5 model. 
5. Adjust the server url, paths, prompts, number of images etc. in `sd_request.py`.
6. Run `sd_request.py`.

If you find our work useful, please consider citing our paper.  
```
@misc{poellabauer2024LUMA++,
      title={Advanced Post-Processing for Object Detection Dataset Generation}, 
      author={Pöllabauer, Thomas and Berkei, Sarah and Knauthe, Volker and Kuijper, Arjan},
      booktitle={19th International Symposium on Visual Computing (ISCV24)},
      year={2024}
}
```  
