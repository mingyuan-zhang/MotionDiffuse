from options.base_options import BaseOptions
import argparse

class TrainCompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--num_layers', type=int, default=8, help='num_layers of transformer')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim of transformer')
        self.parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion_steps of transformer')
        self.parser.add_argument('--no_clip', action='store_true', help='whether use clip pretrain')
        self.parser.add_argument('--no_eff', action='store_true', help='whether use efficient attention')

        self.parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
        self.parser.add_argument('--times', type=int, default=1, help='times of dataset')

        self.parser.add_argument('--feat_bias', type=float, default=25, help='Scales for global motion features and foot contact')

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trail continued from previous trail?')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of saving models (by epoch)')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of animation results (by epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving models (by iteration)')
        self.is_train = True
