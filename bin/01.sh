set -ex

# python examples/train.py --config configs/multiclass.py --fold 0
# python examples/train.py --config configs/multiclass.py --fold 1
# python examples/train.py --config configs/multiclass.py --fold 2
# python examples/train.py --config configs/multiclass.py --fold 3
# python examples/train.py --config configs/multiclass.py --fold 4

python scripts/make_oof.py --version 01 --config configs/multiclass.py