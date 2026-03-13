import numpy as np
import torch

from ..utils import ClipLoss_Modified_DDP


def evaluate_eeg_accuracy(pl_model, test_loader, device):
    """Strictly follow PLModel's test_step logic to evaluate accuracy, adapted to current batch structure, logic fully preserved"""
    print("\n===== Validating EEG Model Accuracy =====")
    # Initialize storage structures
    all_predicted_classes = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
    all_true_labels = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}
    mAP_total = {'image': 0, 'text': 0, 'depth': 0, 'edge': 0, 'fusion': 0}
    match_similarities = {'image': [], 'text': [], 'depth': [], 'edge': [], 'fusion': []}

    # Model setup
    pl_model = pl_model.to(device)
    pl_model.eval()
    pl_model.criterion = ClipLoss_Modified_DDP(top_k=10, cos_batch=512)
    with torch.no_grad():
        for batch in test_loader:
            # 1. Process batch: only move tensors to device, ignore non-tensor data (such as lists, strings)
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # Only process tensor types (data involved in computation)
                    processed_batch[k] = v.to(device, non_blocking=True)
                else:
                    # Non-tensor data (such as paths, text) not moved, keep as is
                    processed_batch[k] = v
            batch = processed_batch
            batch_size = batch['idx'].shape[0]

            # 2. Call model forward (using processed batch)
            eeg_z, mod_z, _ = pl_model(batch)

            # 4. Calculate similarity and Top-5 predictions (reuse test_step logic)
            for key in mod_z:
                similarity = (eeg_z[key] @ mod_z[key].T)
                top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
                all_predicted_classes[key].append(top_k_indices.cpu().numpy())
                all_true_labels[key].extend(batch['label'].cpu().numpy())

                # Collect diagonal similarities
                match_similarities[key].extend(similarity.diag().detach().cpu().tolist())

                # Calculate mAP
                for i in range(similarity.shape[0]):
                    true_index = i
                    sims = similarity[i, :]
                    sorted_indices = torch.argsort(-sims)
                    rank = (sorted_indices == true_index).nonzero()[0][0] + 1
                    mAP_total[key] += 1 / rank

    # 5. Aggregate and calculate accuracy (reuse on_test_epoch_end logic)
    accuracy = {}
    modality_predictions = {}
    modality_labels = {}
    sample_count = None

    for key in all_predicted_classes:
        all_predicted = np.concatenate(all_predicted_classes[key], axis=0)
        all_true = np.array(all_true_labels[key])
        modality_predictions[key] = all_predicted
        modality_labels[key] = all_true

        if sample_count is None:
            sample_count = len(all_true)
        else:
            assert len(all_true) == sample_count, f"Modality {key} sample count inconsistent"
        if key != 'image':
            assert np.all(all_true == modality_labels['image']), f"Modality {key} labels inconsistent"

        # Calculate metrics
        top1_acc = np.mean(all_predicted[:, 0] == all_true)
        top5_acc = np.mean(np.any(all_predicted == all_true[:, np.newaxis], axis=1))
        mAP = mAP_total[key] / len(all_true)
        avg_similarity = np.mean(match_similarities[key]) if match_similarities[key] else 0

        accuracy[key] = {
            'top1': top1_acc,
            'top5': top5_acc,
            'mAP': mAP.item() if isinstance(mAP, torch.Tensor) else mAP,
            'similarity': avg_similarity
        }

        print(f"{key} modality accuracy:")
        print(
            f"  Top-1: {top1_acc:.4f}, Top-5: {top5_acc:.4f}, mAP: {accuracy[key]['mAP']:.4f}, Avg Similarity: {avg_similarity:.4f}")

    # 6. Cross-modal accuracy
    combined_top1 = np.stack([modality_predictions[key][:, 0] for key in modality_predictions], axis=1)
    any_top1_acc = np.mean(np.any(combined_top1 == modality_labels['image'][:, np.newaxis], axis=1))

    combined_top5 = np.stack(
        [modality_predictions[key][:, :5] for key in ['image', 'text', 'depth', 'edge']],
        axis=1
    )
    expanded_labels = modality_labels['image'][:, np.newaxis, np.newaxis]
    modality_top5_correct = np.any(combined_top5 == expanded_labels, axis=2)
    any_top5_acc = np.mean(np.any(modality_top5_correct, axis=1))

    accuracy['cross_modal'] = {
        'top1': any_top1_acc,
        'top5': any_top5_acc
    }
    print("\nCross-modal accuracy:")
    print(f"  Top-1: {any_top1_acc:.4f}, Top-5: {any_top5_acc:.4f}")

    return accuracy
