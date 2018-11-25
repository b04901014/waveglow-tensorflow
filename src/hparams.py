import argparse
import librosa
import numpy as np

parser = argparse.ArgumentParser(description='Tensorflow Implementation of WaveGlow')

##Training Parameters##
#Sizes
parser.add_argument('--batch_size', dest='batch_size', type=int, default=4, help='Batch Size')
parser.add_argument('--mel_time_step', dest='mel_time_step', type=int, default=64, help='Time Step of Inputs')
#Optimizer
parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Initial Learning Rate')
#Epoch Settings
parser.add_argument('--epoch', dest='epoch', type=int, default=100000, help='Number of Epochs')
parser.add_argument('--display_step', dest='display_step', type=int, default=100, help='Batch to Output Training Details')
parser.add_argument('--saving_epoch', dest='saving_epoch', type=int, default=2, help='Epoch to Save Model')
parser.add_argument('--sample_epoch', dest='sample_epoch', type=int, default=1, help='Epoch to Sample')
parser.add_argument('--sample_num', dest='sample_num', type=int, default=5, help='Number of Audios per Sample')
parser.add_argument('--valsplit', dest='valsplit', type=float, default=0.9, help='Portion for training examples, others for validation')
parser.add_argument('--num_proc', dest='num_proc', type=int, default=None, help='Number of process to spawn for data loader')
#GPU
parser.add_argument('--gpu_fraction', dest='gpu_fraction', type=float, default=0.85, help='Fraction of GPU Memory to use')
#parser.add_argument('--use_fp16', dest='use_fp16', default=False, action='store_false', help='True if use float16 for tensorcore acceleration')
#parser.add_argument('--fp16_scale', dest='fp16_scale', type=float, default=128, help='Scaling factor for fp16 computation')
#Normalization
parser.add_argument('--use_weight_norm', dest='use_weight_norm', default=False, action='store_true',  help='Use Weight Normalization or not')
parser.add_argument('--use_instance_norm', dest='use_instance_norm', default=False, action='store_true',  help='Use Instance Normalization or not')

##Inference##
parser.add_argument('--do_infer', dest='is_training', default=True, action='store_false', help='Default to training mode, do inference if --do_infer is specified')
parser.add_argument('--infer_mel_dir', dest='infer_mel_dir', default='/data/lichen/waveglow/training_mels', help='Path to inference numpy files of mel spectrogram')
parser.add_argument('--infer_path', dest='infer_path', default='/data/lichen/waveglow/inference', help='Path to output inference wavs')

##Sampling##
parser.add_argument('--truncate_sample', dest='truncate_sample', default=False, action='store_true', help='Truncate the infer input mels to truncate_step due to GPU memory consideration or not')
parser.add_argument('--truncate_step', dest='truncate_step', type=float, default=384, help='Truncate the infer input mels to truncate_step due to GPU memory consideration')

##Input Path##
parser.add_argument('--metadata_dir', dest='metadata_dir', default='/data/lichen/TTS_data/LJSpeech-1.1/metadata.csv', help='Path to metadata.csv')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='/data/lichen/TTS_data/LJSpeech-1.1/wavs', help='Path to audio file for preprocessing dataset')
parser.add_argument('--mel_dir', dest='mel_dir', default='/data/lichen/waveglow/training_mels', help='Path to input mel spectrogram (Output directory for processing dataset)')
parser.add_argument('--wav_dir', dest='wav_dir', default='/data/lichen/waveglow/training_wavs', help='Path to input audio file for training (Output directory for processing dataset)')

##Output Path##
parser.add_argument('--saving_path', dest='saving_path', default='../model', help='Path to save model if specified')
parser.add_argument('--loading_path', dest='loading_path', default='../model', help='Path to load model if specified')
parser.add_argument('--sampling_path', dest='sampling_path', default='../samples', help='Path to save samples if specified')
parser.add_argument('--summary_dir', dest='summary_dir', default='../summary', help='Path to save summaries')

##Audio Processing Params##
#STFT
parser.add_argument('--num_freq', dest='num_freq', type=int, default=513, help='Number of frequency bins for STFT')
parser.add_argument('--hop_length', dest='hop_length', type=int, default=256, help='Hop length for STFT')
parser.add_argument('--window_size', dest='window_size', type=int, default=1024, help='Window size for STFT')
#Mels
parser.add_argument('--n_mel', dest='n_mel', type=int, default=80, help='Channel Size of Inputs')
parser.add_argument('--fmin', dest='fmin', type=int, default=0, help='Minimum Frequency of Mel Banks')
parser.add_argument('--fmax', dest='fmax', type=int, default=7600, help='Maximum Frequency of Mel Banks')
#Silence
parser.add_argument('--trim_hop_length', dest='trim_hop_length', type=int, default=256, help='Hop length for trimming silence')
parser.add_argument('--trim_window_size', dest='trim_window_size', type=int, default=1024, help='Window size for trimming silence')
parser.add_argument('--trim_inner_scilence', dest='trim_inner_scilence', default=False, action='store_true', help='Specify to trim the inner slience')
#Preprocessing
parser.add_argument('--sample_rate', dest='sample_rate', type=int, default=22050, help='Sample Rate of Input Audios')
parser.add_argument('--trim_top_db', dest='trim_top_db', type=float, default=10, help='Top dB for trimming scilence')
parser.add_argument('--clip_to_value', dest='clip_to_value', type=float, default=4.0, help='Max/Min value of mel spectrogram')
parser.add_argument('--ref_db', dest='ref_db', type=float, default=45, help='Value to subtract to normalize mel spectrogram')
parser.add_argument('--scale_db', dest='scale_db', type=float, default=15, help='Value to divide to normalize mel spectrogram')

##Flow Network##
parser.add_argument('--n_flows', dest='n_flows', type=int, default=12, help='Number of flow layers in network')
parser.add_argument('--squeeze_size', dest='squeeze_size', type=int, default=8, help='Number of channels of input wavs')
parser.add_argument('--early_output_size', dest='early_output_size', type=int, default=2, help='Number of channels per early output')
parser.add_argument('--early_output_every', dest='early_output_every', type=int, default=4, help='Number of flows per early output')
parser.add_argument('--sigma', dest='sigma', type=float, default=1.0, help='Stddev for the gaussian prior during training')
parser.add_argument('--infer_sigma', dest='infer_sigma', type=float, default=0.6, help='Stddev for the gaussian prior during inference')

##WaveNet##
parser.add_argument('--wavnet_channels', dest='wavnet_channels', type=int, default=512, help='Number of WaveNet channels')
parser.add_argument('--wavenet_layers', dest='wavenet_layers', type=int, default=8, help='Number of layers of WaveNet')
parser.add_argument('--wavenet_filter_size', dest='wavenet_filter_size', type=int, default=3, help='Filter length of WaveNet')


args = parser.parse_args()

##The model Structure Hyperparams, NOT CHANGEABLE from command line##
##Params dependent of other params##
args.n_fft = (args.num_freq - 1) * 2
args.melbasis = librosa.filters.mel(args.sample_rate, args.n_fft, n_mels=args.n_mel, fmin=args.fmin, fmax=args.fmax)
#args.melbasisinv = np.linalg.pinv(args.melbasis)
args.step_per_mel = args.hop_length
#args.step_to_scale = args.step_per_mel // args.squeeze_size + 1 if args.step_per_mel % args.squeeze_size else args.step_per_mel // args.squeeze_size
#args.wav_time_step = args.mel_time_step * args.step_to_scale * args.squeeze_size
args.wav_time_step = args.mel_time_step * args.step_per_mel
if args.n_flows % args.early_output_every == 0:
    args.output_remain = args.squeeze_size - (args.n_flows // args.early_output_every - 1) * args.early_output_size
else:
    args.output_remain = args.squeeze_size - (args.n_flows // args.early_output_every) * args.early_output_size

assert args.mel_time_step % args.squeeze_size == 0
assert args.step_per_mel % args.squeeze_size == 0
assert args.n_flows % (args.squeeze_size // args.early_output_size) == 0
print ("Time Segments of audio for training: %d" %args.wav_time_step)
