import numpy as np
import librosa
import os
import sys
import traceback
from tqdm import tqdm
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from hparams import args

#################################################################
# If Hparam adjustion is needed, please change it in hparams.py #
#################################################################

def norm(ifn, ofn_mel, ofn_wav):
    y, _ = librosa.core.load(ifn, sr=args.sample_rate)
    if args.trim_inner_scilence:
        yss = librosa.effects.split(y, top_db=args.trim_top_db, frame_length=args.trim_window_size, hop_length=args.trim_hop_length)
        y = [y[z[0]: z[1] + 10] for z in yss]
        y = np.concatenate(y, axis=0)
    else:
        y, index = librosa.effects.trim(y, top_db=args.trim_top_db, frame_length=args.trim_window_size, hop_length=args.trim_hop_length)
    np.save(ofn_wav, y)
    y = wav2msp(y)
    np.save(ofn_mel, y)

def wav2msp(x):
    ret = librosa.stft(x, n_fft=args.n_fft, hop_length=args.hop_length, win_length=args.window_size)
    ret = np.abs(ret)
    ret = np.matmul(args.melbasis, ret)
    ret = -20 * np.log10(np.maximum(ret, 1e-8))
    ret = (ret - args.ref_db) / args.scale_db
    ret = np.clip(ret, -args.clip_to_value, args.clip_to_value)
    return ret.transpose()

def clean_dir(d):
    for f in os.listdir(d):
        f = os.path.join(op_dir, f)
        if os.path.isfile(f):
            os.remove(f)

def process_one_dir():
    if not os.path.isdir(args.mel_dir):
        print ("Making Directory: %r" %args.mel_dir)
        os.makedirs(args.mel_dir)
    if not os.path.isdir(args.wav_dir):
        print ("Making Directory: %r" %args.wav_dir)
        os.makedirs(args.wav_dir)
#    print ("Cleaning Output Directory %s" %args.mel_dir)
#    clean_dir(args.mel_dir)
#    print ("Cleaning Output Directory %s" %args.wav_dir)
#    clean_dir(args.wav_dir)
    with open(args.metadata_dir, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        wavname, _, text = line.strip().split('|')
        wavpath = os.path.join(args.dataset_dir, wavname) + '.wav'
        ofn_mel = os.path.join(args.mel_dir, wavname)
        ofn_wav = os.path.join(args.wav_dir, wavname)
        try:
            norm(wavpath, ofn_mel, ofn_wav)
#            print ("Writing from %r to %r and %r" %(wavpath, ofn_mel, ofn_wav))
        except:
            traceback.print_exc()
            sys.exit()
#            continue

if __name__ =='__main__':
    process_one_dir()
