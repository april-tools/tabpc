import torch
import sys
import os
import fnmatch
import json

def count_tensors(obj):
    """Recursively count tensor elements in nested structures, excluding integer tensors."""
    if isinstance(obj, torch.Tensor):
        if obj.dtype.is_floating_point or obj.dtype == torch.bool or obj.dtype == torch.complex64 or obj.dtype == torch.complex128:
            return obj.numel()
        else:
            return 0
    elif isinstance(obj, dict):
        return sum(count_tensors(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(count_tensors(v) for v in obj)
    else:
        return 0

def count_parameters_in_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    total_params = count_tensors(state_dict)
    return total_params

def find_files(*, starting_folder: str = ".", pattern: str):
    """
    find all files that match the given pattern, starting from the given folder and going down the directory tree
    """
    matches = []
    for root, _, files in os.walk(starting_folder):
        for filename in files:
            full_name = os.path.join(root, filename)
            if fnmatch.fnmatch(full_name, pattern):
                matches.append(full_name)
    return matches

def printc(message, color):
    """
    Print a message to the terminal in the specified color.

    color: one of "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    """
    colors = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }
    color_code = colors.get(color.lower(), colors["reset"])

    if isinstance(message, dict):
        message = json.dumps(message, indent=4)
    print(f"{color_code}{message}{colors['reset']}")

if __name__ == "__main__":    
    """
    Example usage:
    python number_of_parameters.py <checkpoint_path>
    OR
    python number_of_parameters.py <starting_folder> <pattern>
    OR
    python number_of_parameters.py <starting_folder>
    (in the last case, it will search for *.pt, *.pth, *.ckpt files)
    """

    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if os.path.isdir(arg):
            # If argument is a directory, use default patterns
            patterns = ["*.pt", "*.ckpt", "*.pth"]
            checkpoint_paths = []
            for pattern in patterns:
                checkpoint_paths.extend(find_files(starting_folder=arg, pattern=pattern))
        else:
            checkpoint_paths = [arg]
    elif len(sys.argv) == 3:
        checkpoint_paths = find_files(starting_folder=sys.argv[1], pattern=sys.argv[2])
    else:
        print("Usage: python number_of_parameters.py <checkpoint_path> OR python number_of_parameters.py <starting_folder> <pattern>")
        sys.exit(1)

    for checkpoint_path in checkpoint_paths:
        try:
            print(f"{checkpoint_path}: {count_parameters_in_checkpoint(checkpoint_path)}", "yellow")
        except Exception as e:
            printc(f"Error processing {checkpoint_path}: {e}", "red")
            continue