import argparse

parser = argparse.ArgumentParser([])
parser.add_argument("--model", choices=['MetaCS-Ours', 'Ours'])
parser.add_argument("--device", default='cuda:0')
parser.add_argument("--repeat_times", type=int, default=5)
parser.add_argument("--test", action="store_true", default=False)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--local_lr", type=float, default=2e-3)
parser.add_argument("--local_step", type=int, default=4)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=3)
parser.add_argument("--lambda_", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epoch", type=int, default=50)
parser.add_argument("--embedding_size", type=int, default=32)

args = parser.parse_args()
