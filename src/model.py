import tensorflow as tf
import os
import traceback
from tqdm import tqdm
from hparams import args
from utils import *
from module import *
from ops import *

class WaveGlow():
    def __init__(self, sess):
        self.sess = sess
        self.data_generator = multiproc_reader(500)
        self.validation_data_generator = multiproc_reader_val(args.sample_num * 5)
        self.build_model()

    def build_model(self):
        self.global_step = tf.get_variable('global_step', initializer=0, dtype=tf.int32, trainable=False)
        self.mels = tf.placeholder(tf.float32, [None, args.n_mel, None])
        self.wavs = tf.placeholder(tf.float32, [None, args.squeeze_size, args.wav_time_step // args.squeeze_size])
        self.placeholders = [self.mels, self.wavs]
        self.conditions = mydeconv1d(self.mels, args.n_mel, filter_size=args.step_per_mel*4, stride=args.step_per_mel, scope='upsample', reuse=False)
        self.conditions = tf.transpose(self.conditions, perm=[0, 2, 1])
        self.conditions = tf.reshape(self.conditions, [-1, tf.shape(self.conditions)[1] // args.squeeze_size, args.squeeze_size * args.n_mel])
        self.conditions = tf.transpose(self.conditions, perm=[0, 2, 1])
        self.z = []
        self.layer = self.wavs
        self.logdets, self.logss = 0, 0
        for i in range(args.n_flows):
            self.layer, logs, logdet = conv_afclayer(self.layer, self.conditions, reverse=False, scope='afc_'+str(i+1), reuse=False)
            self.logdets += logdet
            self.logss += logs
            if (i + 1) % args.early_output_every == 0 and (i + 1) != args.n_flows:
                self.z.append(self.layer[:, : args.early_output_size])
                self.layer = self.layer[:, args.early_output_size:]
        self.z.append(self.layer)
        self.z = tf.concat(self.z, axis=1)
        total_size = tf.cast(tf.size(self.z), tf.float32)
        self.logdet_loss = -tf.reduce_sum(self.logdets) / total_size
        self.logs_loss = -tf.reduce_sum(self.logss) / total_size
        self.prior_loss = tf.reduce_sum(self.z ** 2 / (2 * args.sigma ** 2)) / total_size
        self.loss = self.prior_loss + self.logs_loss + self.logdet_loss
        self.t_vars = tf.trainable_variables()
#        print ([v.name for v in self.t_vars])
        self.numpara = 0
        for var in self.t_vars:
            varshape = var.get_shape().as_list()
            self.numpara += np.prod(varshape)
        print ("Total number of parameters: %r" %(self.numpara))

########INFERENCE#########
        self.output = tf.random.truncated_normal([tf.shape(self.conditions)[0], args.output_remain, tf.shape(self.conditions)[2]], dtype=tf.float32)
        for i in reversed(range(args.n_flows)):
            if (i + 1) % args.early_output_every == 0 and (i + 1) != args.n_flows:
                self.newz = tf.random.truncated_normal([tf.shape(self.conditions)[0], args.early_output_size, tf.shape(self.conditions)[2]], stddev=args.infer_sigma)
                self.output = tf.concat([self.newz, self.output], axis=1)
            self.output = conv_afclayer(self.output, self.conditions, reverse=True, scope='afc_'+str(i+1), reuse=tf.AUTO_REUSE)
########INFERENCE#########

    def train(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.grad = self.optimizer.compute_gradients(self.loss, var_list=self.t_vars)
        self.op = self.optimizer.apply_gradients(self.grad, global_step=self.global_step)
        varset = list(set(tf.global_variables()) | set(tf.local_variables()))
        self.saver = tf.train.Saver(var_list=varset, max_to_keep=3)
        num_batch = self.data_generator.n_examples // args.batch_size
        do_initialzie = True
        if args.loading_path:
            if self.load():
                start_epoch = self.global_step.eval() // num_batch
                do_initialzie = False
            else:
                print ("Error Loading Model! Training From Initial State...")
        if do_initialzie:
            init_op = tf.global_variables_initializer()
            start_epoch = 0
            self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.summary_dir, None)
        with tf.name_scope("summaries"):
            self.s_logdet_loss = tf.summary.scalar('logdet_loss', self.logdet_loss)
            self.s_logs_loss = tf.summary.scalar('logs_loss', self.logs_loss)
            self.s_prior_loss = tf.summary.scalar('prior_loss', self.prior_loss)
            self.s_loss = tf.summary.scalar('total_loss', self.loss)
            self.merged = tf.summary.merge([self.s_logdet_loss, self.s_logs_loss, self.s_prior_loss, self.s_loss])

        self.procs = self.data_generator.start_enqueue()
        self.val_procs = self.validation_data_generator.start_enqueue()
        self.procs += self.val_procs

        self.sample(0)
        try:
            for epoch in range(start_epoch, args.epoch):
                loss_names = ["Total Loss",
                              "LogS Loss",
                              "LogDet Loss",
                              "Prior Loss"]
                buffers = buff(loss_names)
                for batch in tqdm(range(num_batch)):
                    input_data = self.data_generator.dequeue()
                    feed_dict = {a: b for a, b in zip(self.placeholders, input_data)}
                    _, loss, logs_loss, logdet_loss, prior_loss, summary, step = self.sess.run([self.op,
                                                                                                self.loss,
                                                                                                self.logs_loss,
                                                                                                self.logdet_loss,
                                                                                                self.prior_loss,
                                                                                                self.merged,
                                                                                                self.global_step],
                                                                                                feed_dict=feed_dict)
                    self.gate_add_summary(summary, step)
                    buffers.put([loss, logs_loss, logdet_loss, prior_loss], [0, 1, 2, 3])
                    if (batch + 1) % args.display_step == 0:
                        buffers.printout([epoch + 1, batch + 1, num_batch])
                if (epoch + 1) % args.saving_epoch == 0 and args.saving_path:
                    try :
                        self.save(epoch + 1)
                    except:
                        print ("Failed saving model, maybe no space left...")
                        traceback.print_exc()
                if (epoch + 1) % args.sample_epoch == 0 and args.sampling_path:
                    self.sample(epoch + 1)
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
        finally:
            for x in self.procs:
                x.terminate()

    def save(self, epoch):
        name = 'Model_Epoch_' + str(epoch)
        saving_path = os.path.join(args.saving_path, name)
        print ("Saving Model to %r" %saving_path)
        step = self.sess.run(self.global_step)
        self.saver.save(self.sess, saving_path, global_step=step)

    def load(self):
        ckpt = tf.train.get_checkpoint_state(args.loading_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print ("Loading Model From %r" %os.path.join(args.loading_path, ckpt_name))
            self.saver.restore(self.sess, os.path.join(args.loading_path, ckpt_name))
            return True
        return False

    def sample(self, epoch):
        print ("Sampling to %r" %args.sampling_path)
        try:
            for i in tqdm(range(args.sample_num)):
                name = 'Epoch_%r-%r.wav' %(epoch, i+1)
                outpath = os.path.join(args.sampling_path, name)
                print ('Sampling to %r ...' %outpath)
                mels = self.validation_data_generator.dequeue()
                output = self.sess.run(self.output, feed_dict={self.mels: [mels]})
                output = np.transpose(output[0])
                output = np.reshape(output, [-1])
                writewav(outpath, output)
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
            for x in self.procs:
                x.terminate()
           
    def infer(self):
        self.infer_data_generator = multiproc_reader_val(100)
        print ("Performing Inference from %r" %args.infer_mel_dir)
        try:
            if self.load():
                step = self.global_step.eval()
            else:
                print ('Error loading model at inference state!')
                raise RuntimeError
            while self.infer_data_generator.alive:
                name = 'Infer_Step_%r-%r.wav' %(step, i+1)
                outpath = os.path.join(args.infer_path, name)
                print ('Synthesizing to %r ...' %outpath)
                mels = self.infer_data_generator.dequeue()
                output = self.sess.run(self.output, feed_dict={self.mels: [mels]})
                output = np.transpose(output[0])
                output = np.reshape(output, [-1])
                writewav(outpath, output)
        except KeyboardInterrupt:
            print ("KeyboardInterrupt")
        except:
            traceback.print_exc()
        finally:
            for x in self.procs:
                x.terminate()
 
    def gate_add_summary(self, summary, step):
        try:      
            self.writer.add_summary(summary, step)
        except:
            print ("Failed adding summary, maybe no space left...")
