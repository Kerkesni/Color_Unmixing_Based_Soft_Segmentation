import argparse
import sys
import os

parser = argparse.ArgumentParser(description='Color Model Estimation')
parser.add_argument('image_path', type=str, nargs='?', help='image path')
parser.add_argument('quality', type=int, nargs='?', default=50, help='image quality level')
args = parser.parse_args()

if(len(sys.argv) < 1):
    print("Not Enough Arguments")
    exit(-1)

print("Color Model Estimation")
os.system(f"distributions.py {args.image_path} {args.quality}")
print("Sparce Color Unmixing")
os.system(f"minimization.py {args.image_path} {args.quality}")
print("Mate Regularization")
os.system(f"mate_regularization.py {args.image_path} {args.quality}")
print("Color Refinement")
os.system(f"color_refinement.py {args.image_path} {args.quality}")

