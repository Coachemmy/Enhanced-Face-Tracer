from argparse import ArgumentParser

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# file paths
		self.parser.add_argument('--dataset_path', type=str, default='', help='dataset path')
		self.parser.add_argument('--log_path', type=str, default='', help='log file path')
		self.parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
		self.parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint file path')
		self.parser.add_argument('--dir_path_root', type=str, default='./path_files/', help='direction path')

		# training/testing related
		self.parser.add_argument('--total_epochs', type=int, default=13, help='total epochs for training')
		self.parser.add_argument('--img_size', type=tuple, default=(112, 112), help='size of the input images')
		self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
		self.parser.add_argument('--quality', type=str, default='new_swaps', help='the quality of input images, selected from c0, c23, c40')
		self.parser.add_argument('--frame_count', type=int, default=1, help='count of the selected in each video after preprocessing')
		self.parser.add_argument('--test_rate', type=int, default=1, help='testing rate during training')
		self.parser.add_argument('--loss_type', type=str, default='aamsoftmax', help='the type of the loss function')
		self.parser.add_argument('--print_rate', type=int, default=100, help='the print rate of the loss function during training')
		self.parser.add_argument('--optimizer', type=str, default='Ranger', help='optimizer of the training process')
		self.parser.add_argument('--scheduler', type=str, default='StepLR', help='scheduler of the training process')
		self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
		self.parser.add_argument('--beta_1', type=float, default=0.95, help='parameter beta_1 of the optimizer')
		self.parser.add_argument('--beta_2', type=float, default=0.999, help='parameter beta_2 of the optimizer')
		self.parser.add_argument('--weight_decay', type=float, default=2e-5, help='parameter weight decay of the optimizer')
		self.parser.add_argument('--step_size', type=int, default=4, help='parameter step size of the scheduler')
		self.parser.add_argument('--gamma', type=float, default=0.1, help='parameter gamma of the scheduler')
		self.parser.add_argument('--k', type=int, default=1, help='k value of top-k acc')
		self.parser.add_argument('--cross', action='store_true', help='whether to do cross test')
		
		# structure/information related
		self.parser.add_argument('--training_mode', type=str, default='dual', help='training modes contains:(1)dual---evidance identity(source) is available (2)single---evidance identity(source) is unavailable')
		self.parser.add_argument('--testing_mode', type=str, default='dual', help='testing modes contains:(1)dual---evidance identity(source) is available (2)single---evidance identity(source) is unavailable')
		self.parser.add_argument('--train_status', type=str, default='train', help='load image training status')
		self.parser.add_argument('--test_status', type=str, default='test', help='load image testing status')
		self.parser.add_argument('--net_type', type=str, default='r18_mix', help='the base disentangle network structure')
		self.parser.add_argument('--train_file_name', type=str, default='full.txt', help='the name of the training file')
		self.parser.add_argument('--test_file_name', type=str, default='rest.txt', help='the name of the cross testing file')
		self.parser.add_argument('--test_origin_file_name', type=str, default='origin.txt', help='the name of the cross testing origin file')

	def parse(self):
		opts = self.parser.parse_args()
		return opts