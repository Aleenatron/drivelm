import argparse
import os
import json
import torch
import pandas as pd
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from tqdm import tqdm as progress_bar
from transformers import T5Tokenizer
from torch.utils.data import DataLoader
from torchvision import transforms

from modules.multi_frame_dataset import MultiFrameDataset
from modules.multi_frame_model import DriveVLMT5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Existing batch validation
# ---------------------------
def val_model(dloader, model, processor, image_id_dict, config):
    model.eval()
    ids_answered = set()
    test_data = []

    with torch.no_grad():
        for idx, (q_texts, encodings, imgs, labels, img_paths) in progress_bar(enumerate(dloader), total=len(dloader)):
            outputs = model.generate(encodings, imgs)
            text_outputs = [processor.decode(output, skip_special_tokens=True) for output in outputs]
            print(f"[INFO] Raw outputs shape: {outputs.shape}")
            print(f"[INFO] Sample decoded output: {text_outputs[0]}")
            print(f"[INFO] Sample question: {q_texts[0]}")
            if idx % 100 == 0:
                print(q_texts)
                print(text_outputs)

            for image_path, q_text, text_output in zip(img_paths, q_texts, text_outputs):
                img_key = image_path[0]
                question_only = q_text.replace("Question: ", "").strip()
                key_string = img_key + ' ' + question_only
                print(f"[DEBUG] Eval key_string: {repr(key_string)}")
                print(f"Present in image_id_dict? {key_string in image_id_dict}")
                if key_string in image_id_dict:
                    print(f"[DEBUG] Image ID: {image_id_dict[key_string][0]}")

                if key_string not in image_id_dict:
                    print(f"[Warning] Missing key in image_id_dict: {key_string}")
                    continue

                if image_id_dict[key_string][0] in ids_answered:
                    continue
                if len(text_output) > config.max_len:
                    continue

                ids_answered.add(image_id_dict[key_string][0])
                test_data.append({'image_id': image_id_dict[key_string][0], 'caption': text_output})

    out_path = os.path.join('multi_frame_results', config.model_name, 'predictions.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(test_data, f)


def save_experiment(coco_eval, config):
    trial_dict = {metric: [score] for metric, score in coco_eval.eval.items()}
    trial_dict = pd.DataFrame(trial_dict)
    out_path = os.path.join('multi_frame_results', config.model_name, 'metrics.csv')
    trial_dict.to_csv(out_path, index=False, header=True)


def params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--gpa-hidden-size", default=128, type=int)
    parser.add_argument("--freeze-lm", action="store_true")
    parser.add_argument("--lm", default="T5-Base", choices=["T5-Base", "T5-Large"], type=str)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-dim", default=64, type=int)
    parser.add_argument("--lora-alpha", default=32, type=int)
    parser.add_argument("--lora-dropout", default=0.05, type=float)
    parser.add_argument("--max-len", default=512, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--model-name", default="T5-Medium", type=str)
    parser.add_argument("--input-json", default=None, type=str, help="Path to custom test json")
    parser.add_argument("--image-id-json", default=None, type=str, help="Path to custom image_id json")
    return parser.parse_args()

# ---------------------------
# NEW: importable helpers
# ---------------------------
def load_components(config):
    """
    Load tokenizer, model, and transform once; reusable for single inference.
    """
    processor = T5Tokenizer.from_pretrained(
        'google-t5/t5-base' if config.lm == 'T5-Base' else 'google-t5/t5-large'
    )
    processor.add_tokens('<')

    model = DriveVLMT5(config).to(device)
    ckpt_path = os.path.join('multi_frame_results', config.model_name, 'latest_model.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])
    return model, processor, tfm


def _build_single_dataloader(temp_json_path, processor, tfm, batch_size=1):
    dset = MultiFrameDataset(
        input_file=temp_json_path,
        tokenizer=processor,
        transform=tfm
    )
    return DataLoader(
        dset, shuffle=False, batch_size=batch_size, drop_last=False,
        collate_fn=dset.test_collate_fn
    )


def run_single_inference(model, processor, tfm, cams_dict, question, image_id_dict, config):
    """
    Reuse the loaded model to answer ONE (cams, question) pair.
    Returns {'image_id': int, 'caption': str} or None.
    """
    # Build a tiny temporary JSON matching your dataset format
    tmp_path = os.path.join('/tmp', 'single_prompt.json')
    os.makedirs('/tmp', exist_ok=True)
    single_prompt = [[
        {"Q": question, "A": "", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None},
        cams_dict
    ]]
    with open(tmp_path, 'w') as f:
        json.dump(single_prompt, f)

    dloader = _build_single_dataloader(tmp_path, processor, tfm, batch_size=1)

    model.eval()
    with torch.no_grad():
        for q_texts, encodings, imgs, labels, img_paths in dloader:
            outputs = model.generate(encodings, imgs)
            text_outputs = [processor.decode(out, skip_special_tokens=True) for out in outputs]
            text_output = text_outputs[0] if text_outputs else ""

            img_key = img_paths[0][0]
            question_only = q_texts[0].replace("Question: ", "").strip()
            key_string = img_key + ' ' + question_only

            if key_string not in image_id_dict:
                print(f"[Warning] Missing key in image_id_dict (single): {key_string}")
                return None

            image_id = image_id_dict[key_string][0]
            if len(text_output) > config.max_len:
                print("[Info] Output exceeds max_len; truncated.")
                text_output = text_output[:config.max_len]

            return {'image_id': image_id, 'caption': text_output}

    return None


def load_image_id_dict(config):
    if config.input_json and "dummyprompts.json" in config.input_json:
        image_id_path = os.path.join('data', 'multi_frame', 'image_id_dummy.json')
    else:
        image_id_path = config.image_id_json or os.path.join('data', 'multi_frame', 'image_id.json')
    with open(image_id_path) as f:
        return json.load(f)

# ---------------------------
# CLI entry (unchanged batch flow)
# ---------------------------
if __name__ == "__main__":
    config = params()

    # Components
    model, processor, tfm = load_components(config)

    # Dataset (batch)
    input_file = os.path.join('data', 'multi_frame', 'dummyprompts.json') \
        if config.input_json is None else config.input_json
    test_dset = MultiFrameDataset(
        input_file=input_file,
        tokenizer=processor,
        transform=tfm
    )
    test_dloader = DataLoader(
        test_dset, shuffle=False, batch_size=config.batch_size, drop_last=False,
        collate_fn=test_dset.test_collate_fn
    )

    # image_id mapping
    image_id_dict = load_image_id_dict(config)

    # Run inference
    val_model(test_dloader, model, processor, image_id_dict, config)

    # Optional: COCO eval (kept commented as in your file)
    annotation_file = os.path.join('data', 'multi_frame', 'multi_frame_test_coco.json')
    results_file = os.path.join('multi_frame_results', config.model_name, 'predictions.json')
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()
    save_experiment(coco_eval, config)

