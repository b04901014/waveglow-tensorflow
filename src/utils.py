import numpy as np
import random
import os
import multiprocessing
import librosa
from hparams import args
from scipy.io.wavfile import read, write

def writewav(path, arr):
    librosa.output.write_wav(path, arr, sr=args.sample_rate)

def padtomaxlen(mel, wav):
    if mel.shape[0] > args.mel_time_step:
        rn = random.randint(0, mel.shape[0] - args.mel_time_step)
        wav_idx = rn * args.step_per_mel
        if wav_idx + args.wav_time_step > len(wav):
            wav_idx = len(wav) - args.wav_time_step
        mel = mel[rn: rn + args.mel_time_step]
        wav = wav[wav_idx: wav_idx + args.wav_time_step]
    if mel.shape[0] < args.mel_time_step:
        mel_pad = np.zeros([args.mel_time_step - mel.shape[0]] + list(mel.shape[1:])) - args.clip_to_value
        mel = np.concatenate([mel, mel_pad], axis=0)
    else:
        mel = mel[: args.mel_time_step]
    if wav.shape[0] < args.wav_time_step:
        wav_pad = np.zeros([args.wav_time_step - wav.shape[0]] + list(wav.shape[1:]))
        wav = np.concatenate([wav, wav_pad], axis=0)
    else:
        wav = wav[: args.wav_time_step]
    assert mel.shape[0] == args.mel_time_step
    assert wav.shape[0] == args.wav_time_step
    
    return mel, wav
            
class multiproc_reader():
    def __init__(self, qsize):
        self.manager = multiprocessing.Manager()
        self.queue = multiprocessing.Queue(qsize)
        self.lock = multiprocessing.Lock()
        self.cnt = multiprocessing.Value('i', 0)
        self.load_metadata()
        self.metadata = self.manager.list(self.metadata)
        random.shuffle(self.metadata)

    def load_metadata(self):
        with open(args.metadata_dir, 'r') as f:
            self.metadata = [line.strip().split('|') for line in f]
        self.metadata = self.metadata[: int(len(self.metadata) * args.valsplit)]
        self.n_examples = len(self.metadata)
        print ("Total number of audio/text pair for training: %r" %self.n_examples)

    def dequeue(self):
        return self.queue.get()

    def main_proc(self, cnt):
        while True:
            mels, wavs = [], []
            for _ in range(args.batch_size):
                self.lock.acquire()
                c = cnt.value
                cnt.value += 1
                if cnt.value >= self.n_examples:
                    random.shuffle(self.metadata)
                    cnt.value = 0
                self.lock.release()
                cmetadata = self.metadata[c]
                name, _, text = cmetadata
                melname = os.path.join(args.mel_dir, name) + '.npy'
                wavname = os.path.join(args.wav_dir, name) + '.npy'
                mel, audio = padtomaxlen(np.load(melname), np.load(wavname))
                mels.append(mel)
                wavs.append(audio)
            mels = np.transpose(np.array(mels), axes=[0, 2, 1])
            wavs = np.reshape(np.array(wavs), [args.batch_size, args.wav_time_step // args.squeeze_size, args.squeeze_size])
            wavs = np.transpose(wavs, axes=[0, 2, 1])
            self.queue.put([mels, wavs])

    def start_enqueue(self, num_proc=multiprocessing.cpu_count() // 2):
        if args.num_proc is not None:
            num_proc = args.num_proc
        procs = []
        for _ in range(num_proc):
            p = multiprocessing.Process(target=self.main_proc, args=(self.cnt,))
            p.start()
            procs.append(p)
        return procs

    def printqsize(self):
        print ("Queue Size : ", self.queue.qsize())

class multiproc_reader_val(multiproc_reader):
    def load_metadata(self):
        with open(args.metadata_dir, 'r') as f:
            self.metadata = [line.strip().split('|') for line in f]
            self.metadata = self.metadata[int(len(self.metadata) * args.valsplit): ]
        self.n_examples = len(self.metadata)
        print ("Total number of audio/text pair for validation: %r" %self.n_examples)

    def main_proc(self, cnt):
        while True:
            self.lock.acquire()
            c = cnt.value
            cnt.value += 1
            if cnt.value >= self.n_examples:
                random.shuffle(self.metadata)
                cnt.value = 0
            self.lock.release()
            cmetadata = self.metadata[c]
            name, _, text = cmetadata
            melname = os.path.join(args.mel_dir, name) + '.npy'
            mel = np.load(melname)
            #truncate the samples due to GPU memory consideration
            if args.truncate_sample:
                mel = mel[: args.truncate_step]
            mel = np.transpose(mel)
            self.queue.put(mel)

class multiproc_reader_infer(multiproc_reader_val):
    def __init__(self, qsize):
        self.manager = multiprocessing.Manager()
        self.queue = multiprocessing.Queue(qsize)
        self.lock = multiprocessing.Lock()
        self.cnt = multiprocessing.Value('i', 0)

    def main_proc(self, cnt):
        self.alive = True
        for name in os.path.listdir(args.infer_mel_dir):
            if os.path.splitext(name)[1] != '.npy':
                continue
            melname = os.path.join(args.infer_mel_dir, name)
            mel = np.load(melname)
            mel = np.transpose(mel)
            self.queue.put(mel)
        self.alive = False
