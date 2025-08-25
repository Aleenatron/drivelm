# EM-VLM4AD

## Installation
1. Clone this repository
2. In the repository directory, run `mkdir multi_frame_results`
3. To replicate our environment use the `env.yml` we have provided. The following commands should create a proper environment:
```
conda env create -f env.yml
conda activate EM-VLM4AD
```
## Model Weights
* You can download the model weights for the [T5-Base](https://drive.google.com/drive/folders/1K61Ou-m5c5UmN2ggT-Huw3rv7PhW5Wft?usp=sharing) and [T5-Large-Q](https://drive.google.com/drive/folders/12bHyRTpWWxIJ2pb0WWzfX5mMdkNHKMVP?usp=sharing) version of EM-VLM4AD at the following links. Put the folders for each of these models into the `multi_frame_results` folder. Your directory should look like the following:
```
└── rootFolder
 ├── multi_frame_results/
      ├── T5-Medium/
        ├── latest_model.pth
      ├── T5-Large/
        ├── latest_model.pth
```
## Dataset
First download the train/val/test split [here](https://drive.google.com/file/d/1isiXXTg46nl5SqMiEV4XjFD71KCCzezi/view?usp=sharing) in your root folder. This will include data from the DriveLM dataset as well as the train/val/test splits we use for our experiments. The folder structure should now be as follows: 
```
└── rootFolder
  ├── data/
    ├── multi_frame/
      ├── multi_frame_train.json
      ├── multi_frame_val.json
      ├── multi_frame_test.json
      ├── multi_frame_test_coco.json
      ├── image_id.json
    ├── QA_dataset_nus/
      ├── v1_0_train_nus.json
    ├── nuscenes/
      ├── samples/
  ├── multi_frame_results/
      ├── T5-Medium/
      ├── T5-Large/
```
## Training
* To run training, run `python train.py --batch-size [BATCH SIZE] --epochs [EPOCHS] --lm {T5-Base, T5-Large}`. For more information on other hyperparameters such as loading checkpoints or altering learning rate, weight decay, or the hidden size for gated pooling attention, run `python train.py --help`.
## Inference
* For inference to generate BLEU-4, CIDEr, METEOR, and ROUGE_L metrics for trained models, you can run `python eval.py --batch-size [BATCH_SIZE] --lm {T5-Base, T5-Large} --checkpoint-file [CHECKPOINT_FOLDER]`. For more information on other hyperparameters to work for different model configurations, run `python eval.py --help`.
* We use the [pycocoevalcap](https://github.com/salaniz/pycocoevalcap) library to generate the caption metrics we evaluate on. For this library, Java needs to be installed on your computer. We also recommend commenting out [this line](https://github.com/salaniz/pycocoevalcap/blob/master/eval.py#L45) from the pycocoevalcap library to avoid generating SPICE metrics, which can take longer and don't work for multi-frame situations. 
  ```
python eval.py \
  --batch-size 4 \
  --lm T5-Base \
  --model-name T5-Medium \
  --input-json data/multi_frame/dummyprompts.json \
  --image-id-json data/multi_frame/image_id_dummy.json \
  --annotation-file data/multi_frame/dummy_multi_frame_test_coco.json
  ```
## Running Streamlit Apps

* To run **DriveLM and the annotation tool**, use:
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

* To run **evaluation metrics for the user study**, use:
\`\`\`bash
streamlit run final_streamlit_eval_dashboard.py
\`\`\`

