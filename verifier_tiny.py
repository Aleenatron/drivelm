import json

pred_path = "./multi_frame_results/T5-Medium/predictions.json"
gt_path   = "data/multi_frame/dummy_multi_frame_test_coco.json"

with open(pred_path) as f:
    preds = json.load(f)
with open(gt_path) as f:
    gt = json.load(f)

pred_ids = sorted(set(int(x["image_id"]) for x in preds))
gt_ids = sorted(set(int(img["id"]) for img in gt["images"]))

print("Pred IDs:", pred_ids)
print("GT IDs:", gt_ids)
print("Extra in preds:", set(pred_ids) - set(gt_ids))
print("Extra in GT:", set(gt_ids) - set(pred_ids))

# 💡 Optionally auto-clean predictions
if set(pred_ids) - set(gt_ids):
    preds = [p for p in preds if int(p["image_id"]) in gt_ids]
    with open(pred_path.replace(".json","_clean.json"), "w") as f:
        json.dump(preds, f, indent=2)
    print("Saved cleaned predictions:", pred_path.replace(".json","_clean.json"))


