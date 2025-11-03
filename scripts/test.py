import os
import sys
import time
import torch
import warnings
sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from training.coach import Coach

warnings.filterwarnings('ignore')

def main(opts):
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	torch.autograd.set_detect_anomaly(True)

	coach = Coach(opts)
	os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

	coach.load_model(opts.model_path)
	n_epoch = 0
	coach.cross_test(n_epoch)

if __name__ == '__main__':
	opts = TestOptions().parse()
	main(opts)