import os
import argparse

### CONFIGURATIONS
parser = argparse.ArgumentParser()

# Training Parameters
parser.add_argument("--train-type", type=str, default="supervised")  # ssl
parser.add_argument("--train-mode", type=str, default="regression", choices=["regression", "binary_class"])
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--scheduler", type=str, default="poly", choices=["poly", "cos"])
parser.add_argument('--lr-sch-start', type=int, default=0)
parser.add_argument('--warmup-iters', type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--decay", type=bool, default=True)
parser.add_argument("--decay-rate", type=int, default=0.1)
parser.add_argument("--decay-iter", type=int, default=56000)

# Data Parameters
parser.add_argument("--normalize-label", default=False, action="store_true")  # used to normalize labels (pcwp)
parser.add_argument("--label", type=str, default="pcwp", choices=["pcwp", "age", "gender"])
parser.add_argument("--pcwp-th", type=int, default=18)
parser.add_argument('--num-classes', type=int, default=1)
parser.add_argument("--dir-csv", type=str, default='/storage/shared/apollo/same-day/')

# Model Parameters
parser.add_argument("--model", type=str, default="cnn")  # model name
parser.add_argument("--pretrain", default=False, action="store_true")
parser.add_argument("--load-model", default=False, action="store_true")
parser.add_argument('--load-epoch', type=int, default=None)

# Architecture Parameters
parser.add_argument("--num-layers", type=int, default=2)
parser.add_argument("--input-dim", type=int, default=64)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--embedding-dim", type=int, default=256)

args = parser.parse_args()

# # Device Settings
# if args.device is not None:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     device = str(args.device[0])
#     for i in range(len(args.device) - 1):
#         device += "," + str(args.device[i + 1])
#     os.environ["CUDA_VISIBLE_DEVICES"] = device
