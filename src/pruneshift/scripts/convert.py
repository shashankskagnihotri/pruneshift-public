import argparse
import torch

from pruneshift.utils import load_state_dict


parser = argparse.ArgumentParser(description="Transforms state dicts")
parser.add_argument("input_file", type=str, nargs=1)
parser.add_argument("output_file", type=str, nargs=1)
args = parser.parse_args()


def convert():
    in_path, out_path = args.input_file[0], args.output_file[0]
    print(f"Converting from {in_path} to {out_path}")
    state_dict = load_state_dict(in_path)
    torch.save(state_dict, out_path)


if __name__ == "__main__":
    convert()
