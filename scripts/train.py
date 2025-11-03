import os
import sys
import time
import torch
import warnings
sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from training.coach import Coach

warnings.filterwarnings('ignore')

def main(opts):
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True
	torch.autograd.set_detect_anomaly(True)
	os.makedirs(opts.log_path, exist_ok=True)

	coach = Coach(opts)

	if opts.resume:
		epoch_0 = coach.load_checkpoint(opts.checkpoint_path)
	else:
		epoch_0 = 0
	
	for n_epoch in range(epoch_0, opts.total_epochs):
		with torch.autograd.detect_anomaly():
			coach.train(n_epoch)
		if (n_epoch + 1) % opts.test_rate == 0:
			if opts.cross:
				coach.cross_test(n_epoch)
			coach.save_checkpoint(n_epoch, opts.log_path)
	
	coach.save_model(opts.log_path)

if __name__ == '__main__':
	opts = TrainOptions().parse()
	main(opts)