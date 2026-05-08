import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    length = torch.tensor([elem["data_object"].shape[-1] for elem in dataset_items])
    max_len = length.max().item()
    new_dataset_items = []
    for elem in dataset_items:
        cur_len = elem["data_object"].shape[-1]
        new_elem = F.pad(elem["data_object"], (0, max_len - cur_len))
        new_dataset_items.append(new_elem)
    result_batch["data_object"] = torch.stack(
        [elem for elem in new_dataset_items], dim=0
    )
    result_batch["original_length"] = length
    result_batch["length"] = torch.tensor(
        [elem.shape[-1] for elem in new_dataset_items]
    )
    result_batch["original_sample_rate"] = torch.tensor(
        [elem["original_sample_rate"] for elem in dataset_items]
    )
    result_batch["sample_rate"] = torch.tensor(
        [elem["sample_rate"] for elem in dataset_items]
    )

    return result_batch
