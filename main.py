import argparse
from dataset import deepKNetDataset

def main():
    dataset = deepKNetDataset('./data/', 'band_gap')


if __name__ == "__main__":
    main()


