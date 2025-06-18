from util.box_ops import rescale_bboxes
from PIL import Image
import pdb
import argparse
import json
import numpy as np
import torch
from glob import glob
from model.deformable_detr import DeformableDetrConfig, DeformableDetrFeatureExtractor
from model.egtr import DetrForSceneGraphGeneration
from data.visual_genome import VGDataset
from lib.pytorch_misc import argsort_desc
from transformers.models.detr.feature_extraction_detr import (
    center_to_corners_format,
)

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super(NumpyEncoder, self).default(o)


def process_bbox(out_bbox, orig_size):
    "Take bbox logits and return the image bbox to corner's format"

    box = center_to_corners_format(out_bbox)
    img_h= orig_size[0]
    img_w= orig_size[1]

    scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(box.device)
    box = box * scale_fct  # Simple broadcasting will work for a single image
    pdb.set_trace()
    return box


def parse_arguments():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument(
        "--artifact_path",
        type=str,
        default="./artifacts",
    )
    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10)
    # Evaluation
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_single_preds", type=str2bool, default=True)
    parser.add_argument("--eval_multiple_preds", type=str2bool, default=False)
    parser.add_argument("--logit_adjustment", type=str2bool, default=False)
    parser.add_argument("--logit_adj_tau", type=float, default=0.3)
    # FPS
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--max_size", type=int, default=1333)
    parser.add_argument("--infer_only", type=str2bool, default=False)
    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    # Visualization
    parser.add_argument(
        "--image_path", type=str, default="./test.jpg", help="Input image path"
    )
    parser.add_argument(
        "--output_json", type=str, default="scene_graph.json", help="Output JSON path"
    )
    parser.add_argument(
        "--object_threshold",
        type=float,
        default=0.5,
        help="Object detection confidence threshold",
    )
    parser.add_argument(
        "--connectivity_threshold",
        type=float,
        default=0.5,
        help="Relationship connectivity threshold",
    )
    parser.add_argument(
        "--predicate_threshold",
        type=float,
        default=0.3,
        help="Predicate confidence threshold",
    )
    return parser


def main():
    parser = parse_arguments()
    args, _ = parser.parse_known_args()
    # Load model
    config = DeformableDetrConfig.from_pretrained(args.artifact_path)
    config.logit_adjustment = args.logit_adjustment
    config.logit_adj_tau = args.logit_adj_tau
    model = DetrForSceneGraphGeneration.from_pretrained(
        args.architecture, config=config, ignore_mismatched_sizes=True
    )

    # Load checkpoint
    ckpt_path = sorted(
        glob(f"{args.artifact_path}/checkpoints/epoch=*.ckpt"),
        key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
    )[-1]
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        state_dict[k[6:]] = state_dict.pop(k)  # Remove "model." prefix

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        "SenseTime/deformable-detr", size=args.min_size, max_size=args.max_size
    )

    # Preprocess image
    image = Image.open(args.image_path).convert("RGB")

    # note that is padding the images with the largest image in the batch (here we have 1 image onlx
    pixel_values = feature_extractor(images=image, return_tensors="pt").to(device)
    if torch.cuda.is_available():
        pixel_values["pixel_values"] = pixel_values["pixel_values"].cuda()
        pixel_values["pixel_mask"] = pixel_values["pixel_mask"].cuda()

    # Run inference
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values["pixel_values"],
            pixel_mask=pixel_values["pixel_mask"],
            output_attentions=False,
            output_attention_states=True,
            output_hidden_states=True,
        )

    # Get original image size
    orig_size = torch.tensor([image.height, image.width]).unsqueeze(0).to(device)

    # Load dataset to get label mappings
    test_dataset = VGDataset(
        data_folder=args.data_path,
        feature_extractor=feature_extractor,
        split=args.split,
        num_object_queries=args.num_queries,
    )

    # Create label mappings
    id2label = {k - 1: v["name"] for k, v in test_dataset.coco.cats.items()}

    rel_categories = test_dataset.rel_categories
    num_labels = max(id2label.keys()) + 1
    # Object scores and classes
    pred_logits = outputs["logits"][0]
    pred_boxes = outputs.pred_boxes[0]

    obj_scores, pred_classes = torch.max(pred_logits.softmax(-1)[:, :num_labels], -1)
    pred_classes = pred_classes.clone().cpu().numpy().tolist()

    pdb.set_trace()
    bboxes = rescale_bboxes(pred_boxes.cpu(), (image.width,image.height))

    sub_ob_scores = torch.outer(obj_scores, obj_scores)
    sub_ob_scores[
        torch.arange(pred_logits.size(0)), torch.arange(pred_logits.size(0))
    ] = 0.0  # prevent self-connection

    # predicate score with connectivity
    pred_connectivity = torch.clamp(outputs["pred_connectivity"], 0.0, 1.0)
    # clamp connectivity scores and weigh them with relationship scores
    pred_rel = torch.clamp(
        outputs["pred_rel"], 0.0, 1.0
    )  # Shape: [num_queries, num_queries, num_rel_classes]
    pred_rel = torch.mul(
        pred_rel, pred_connectivity
    )  # Shape: [num_queries, num_queries, 1]

    triplet_score = torch.mul(
        pred_rel.max(-1)[0], sub_ob_scores
    )  # element-wise multiplication
    # Get the object indices of the top 10 triplets based on the triplet score
    pred_rel_inds = argsort_desc(triplet_score.cpu().clone().numpy())[
        : args.topk, :
    ]  # [pred_rels, 2(s,o)]
    # can be used to depict histograms for each predicate of the selected subjects/objects
    rel_scores = (
        pred_rel.cpu()
        .clone()
        .numpy()[pred_rel_inds[:, 0], pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
    )  # [pred_rels, topk]

    best_scores = (
        pred_rel.max(-1)[0]
        .squeeze()[pred_rel_inds[:, 1], pred_rel_inds[:, 2]]
        .cpu()
        .numpy()
        .tolist()
    )
    best_predicate_idx = pred_rel.max(-1)[1].squeeze()[
        pred_rel_inds[:, 1], pred_rel_inds[:, 2]
    ]
    best_predicate_indices = best_predicate_idx.cpu().numpy().tolist()

    # find true names and bbs
    predicates = [rel_categories[i] for i in best_predicate_indices]
    subject_names = [id2label[pred_classes[i]] for i in pred_rel_inds[:, 1]]

    subject_bbox = bboxes.squeeze(dim=0)[pred_rel_inds[:, 1]].cpu().clone().numpy().tolist()
    object_names = [id2label[pred_classes[j]] for j in pred_rel_inds[:, 2]]
    object_bbox = bboxes.squeeze(dim=0)[pred_rel_inds[:, 2]].cpu().clone().numpy().tolist()

    pdb.set_trace()
    # Format triplets
    triplets = []
    for i in range(len(pred_rel_inds)):
        triplets.append(
            {
                "subject": subject_names[i],
                "sub_bbox": subject_bbox[i],
                "predicate": predicates[i],
                "object": object_names[i],
                "obj_bbox": object_bbox[i],
                "score": best_scores[i],
            }
        )
    # Save results to JSON
    with open(args.output_json, "w") as f:
       json.dump(triplets, f, indent=4, cls=NumpyEncoder)

    image_np = np.array(image)

    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    #Add bounding box (using first box)
    x1, y1, x2, y2 = triplets[0]["obj_bbox"]
    pdb.set_trace()
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                           linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()


if __name__ == "__main__":
    main()
