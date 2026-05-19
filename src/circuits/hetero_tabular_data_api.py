from typing import List

import pandas as pd
import torch
from torch import Tensor


def auto_input_layers(
    x: Tensor, metadata: dict, verbose: bool = True
) -> List[dict[str, dict]]:
    """
    Automatically determine input layers based on the DataFrame's columns.

    Args:
        x (Tensor): A Torch tensor representing the input data.

    Returns:
        List[dict[str, Any]]: A list of dictionaries specifying input layers.
    """

    input_layers = {}
    metadata = pd.Series(metadata)

    for col_index in range(x.shape[1]):
        col_index = int(col_index)
        type_name = metadata.iloc[col_index]
        col = x[:, col_index]

        if "cat" in type_name:
            input_layers[col_index] = {
                "name": "categorical",
                "args": {"num_categories": len(torch.unique(col))},
            }
        elif "int" in type_name:
            input_layers[col_index] = {
                "name": "discretized_logistic",
                "args": {
                    "marginal_mean": col.float().mean().item(),
                    "marginal_stddev": col.float().std().item(),
                },
            }
        else:
            input_layers[col_index] = {"name": "gaussian", "args": {}}

    if verbose:
        import json

        print(f"Auto-detected input layers:")
        input_layers_with_info = {
            i: {"col_name": metadata.index[i]} | input_layers[i] for i in input_layers
        }
        print(json.dumps(input_layers_with_info, indent=2))

    return list(input_layers.values())
