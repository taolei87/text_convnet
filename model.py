import os, sys, random, argparse, time, math, gzip
import cPickle as pickle
from collections import Counter

import numpy, numpy.random
import theano
import theano.tensor as T

numpy.set_printoptions(precision=3)

def say(s):
    print "{0}".format(s)
    sys.stdout.flush()

class Pipe:

    @staticmethod
    def create_vocabulary(args):
        cntx = Counter()
        cnty = Counter()
        with open(args.train) as fin:
            lines = fin.readlines()
            raw = [ Pipe.process_line(l) for l in lines ]
        cntx = Counter( [ w for e in raw for w in e[0] ] )
        cnty = Counter( [ e[1] for e in raw ] )
        lst = [ x for x, y in cntx.iteritems() if y > args.cutoff ] + ["## UNK ##"]
        vocabx = dict([ (y,x) for x,y in enumerate(lst) ])
        vocaby = dict([ (y,x) for x,y in enumerate(cnty.keys()) ])
        say( "%d unique words, %d unique labels" % (len(vocabx), len(vocaby)) )
        return vocabx, vocaby

    @staticmethod
    def load_embeddings(args):
        lst = [ ]
        vas = [ ]
        with gzip.open(args.embedding) as fin:
            for line in fin:
                parts = line.strip().split()
                w = parts[0]
                e = numpy.array( [[ float(x) for x in parts[1:] ]],
                                 dtype = theano.config.floatX )
                lst.append(w)
                vas.append(e)
        lst.append("## UNK ##")
        vas.append( numpy.zeros(vas[0].shape, dtype = theano.config.floatX) )
        vocabx = dict([ (y,x) for x,y in enumerate(lst) ])
        embeddings = numpy.concatenate(vas)
        assert len(vocabx) == len(embeddings)
        print "{} embedding loaded, size {}".format(embeddings.shape[0], embeddings.shape[1])
        return vocabx, embeddings

    @staticmethod
    def process_line(line):
        parts = line.strip().split()
        y = parts[0]
        x = parts[1:]
        return x,y

    @staticmethod
    def map_to_idx(x, vocabx, args):
        return numpy.array(
                [ vocabx[w] if w in vocabx else len(vocabx)-1
                    for w in x  ],
                dtype = "int32"
            )

    @staticmethod
    def read_corpus(path, args, vocabx, vocaby):
        corpus_x = [ ]
        corpus_y = [ ]
        with open(path) as fin:
            lines = fin.readlines()
            raw = [ Pipe.process_line(l) for l in lines ]
            raw = [ x for x in raw if len(x[0]) ]
        corpus_x = [ Pipe.map_to_idx(e[0], vocabx, args) for e in raw ]
        corpus_y = [ vocaby[e[1]] for e in raw ]
        oov = [ 1.0 if w not in vocabx else 0 for e in raw for w in e[0] ]
        print "{}: size={}, oov rate={}".format(
                path, len(corpus_x), sum(oov)/(len(oov)+0.0)
            )
        return corpus_x, corpus_y



# Rectified linear unit ReLU(x) = max(0, x)
ReLU = lambda x: x * (x > 0)


# typical output layer is a softmax
class SoftmaxLayer(object):

    def __init__(self, input, n_in, n_out, W=None):

        self.input = input

        if W is None:
            self.W = theano.shared(
                    value = numpy.zeros(
                        (n_in, n_out),
                        dtype = theano.config.floatX),
                    name = 'W',
                    borrow = True
                )
        else:
            self.W = W

        self.s_y_given_x = T.dot(input, self.W)
        self.p_y_given_x = T.nnet.softmax(self.s_y_given_x) #+ self.b)
        self.pred = T.argmax(self.s_y_given_x, axis=1)

        self.params = [ self.W ]

    def nll_loss(self, y):
        return - T.mean(T.log(self.p_y_given_x[T.arange(y.shape[0]),y]))
        #return -T.log(self.p_y_given_x[0,y])

class StrConvLayer:

    def __init__(self, rng, input, n_in, n_out, decay, activation=None, order=3,
            P=None, Q=None, R=None, O=None, b=None):
        # input is len*batch*n_in
        # output is len*batch*n_out

        self.input = input
        self.decay = decay
        self.n_in = n_in
        self.n_out = n_out

        if P is None:
            P_values = numpy.asarray(
                    rng.uniform(
                         low = -(3.0/n_out)**0.5,
                         high = (3.0/n_out)**0.5,
                         size = (n_in, n_out)
                    ),
                    dtype = theano.config.floatX)
            self.P = theano.shared(value = P_values, name = "P", borrow = True)
        else:
            self.P = P

        if Q is None:
            Q_values = numpy.asarray(
                    rng.uniform(
                         low = -(3.0/n_out)**0.5,
                         high = (3.0/n_out)**0.5,
                         size = (n_in, n_out)
                    ),
                    dtype = theano.config.floatX)
            self.Q = theano.shared(value = Q_values, name = "Q", borrow = True)
        else:
            self.Q = Q

        if R is None:
            R_values = numpy.asarray(
                    rng.uniform(
                         low = -(3.0/n_out)**0.5,
                         high = (3.0/n_out)**0.5,
                         size = (n_in, n_out)
                    ),
                    dtype = theano.config.floatX)
            self.R = theano.shared(value = R_values, name = "R", borrow = True)
        else:
            self.R = R

        if O is None:
            O_values = numpy.asarray(
                    rng.uniform(
                         low = -(3.0/n_out)**0.5,
                         high = (3.0/n_out)**0.5,
                         size = (n_out, n_out)
                    ),
                    dtype = theano.config.floatX)
            self.O = theano.shared(value = O_values, name = "O", borrow = True)
        else:
            self.O = O

        if activation != None and activation.lower() == "relu":
            if b is None:
                b_values = numpy.zeros( (self.n_out,), dtype = theano.config.floatX ) + 0.01
                self.b = theano.shared(value = b_values, name = 'b', borrow = True)
            else:
                self.b = b
        elif activation != None and activation.lower() == "tanh":
            if b is None:
                b_values = numpy.zeros( (self.n_out,), dtype = theano.config.floatX )
                self.b = theano.shared(value = b_values, name = 'b', borrow = True)
            else:
                self.b = b
        else:
            self.b = None

        f1, s1, f2, s2, f3 = self.calculate(input)

        if order == 3:
            self.params = [ self.P, self.Q, self.R, self.O ]
            self.output = self.apply_activation(T.dot(f1+f2+f3, self.O), activation)
        elif order == 2:
            self.params = [ self.P, self.Q, self.O ]
            self.output = self.apply_activation(T.dot(f1+f2, self.O), activation)
        elif order == 1:
            self.params = [ self.P, self.O ]
            self.output = self.apply_activation(T.dot(f1, self.O), activation)
        else:
            raise ValueError(
                    "unsupported order: {}".format(order)
                )

        if not (self.b is None):
            self.params.append(self.b)

    def calculate(self, input):
        P = self.P
        Q = self.Q
        R = self.R
        decay = self.decay
        f_0 = T.zeros((input.shape[1], self.n_out), dtype=theano.config.floatX)
        ([f1, s1, f2, s2, f3], updates) = theano.scan( fn = StrConvLayer.loop_one_step,
                sequences = input,
                outputs_info = [ f_0, f_0, f_0, f_0, f_0 ],
                non_sequences = [ P, Q, R, decay ]
            )
        return f1, s1, f2, s2, f3


    # ###
    # Dynamic programming to calculate aggregated unigram to trigram representation vectors
    # ###
    @staticmethod
    def loop_one_step(x_t, f1_tm1, s1_tm1, f2_tm1, s2_tm1, f3_tm1, P, Q, R, decay):
        f1_t = T.dot(x_t, P)
        s1_t = s1_tm1 * decay + f1_t
        f2_t = T.dot(x_t, Q) * s1_tm1
        s2_t = s2_tm1 * decay + f2_t
        f3_t = T.dot(x_t, R) * s2_tm1
        return f1_t, s1_t, f2_t, s2_t, f3_t


    def apply_activation(self, f, activation):
        if activation is None:
            return f
        elif activation.lower() == "none":
            return f
        elif activation.lower() == "relu":
            return ReLU(f+self.b)
        elif activation.lower() == "tanh":
            return T.tanh(f+self.b)
        else:
            raise ValueError(
                    "unsupported activation: {}".format(activation)
                )


class ConvModel:
    def __init__(self, vocabx, vocaby, args):
        self.vocabx = vocabx
        self.vocaby = vocaby
        self.args = args

    def ready(self, embeddings):
        args = self.args
        self.n_hidden = args.hidden_dim
        self.n_in = self.n_hidden if embeddings is None else embeddings.shape[1]
        rng = numpy.random.RandomState(random.randint(0, 9999))

        # x is length * batch_size
        # y is batch_size
        self.x = T.imatrix('x')
        self.y = T.ivector('y')

        x = self.x
        y = self.y
        n_hidden = self.n_hidden
        n_in = self.n_in
        vocabx = self.vocabx
        vocaby = self.vocaby

        # |vocabx| * d
        if embeddings is None:
            EMB_values = numpy.asarray(
                    rng.uniform(
                        low = -(3.0/n_hidden)**0.5,
                        high = (3.0/n_hidden)**0.5,
                        size = (len(vocabx), n_in)
                    ),
                    dtype = theano.config.floatX)
        else:
            EMB_values = embeddings
        print EMB_values.shape
        self.EMB = theano.shared(value = EMB_values, name = "EMB", borrow = True)
        EMB = self.EMB

        # fetch word embeddings
        # (len * batch_size) * n_in
        slices  = EMB[x.flatten()]
        self.slices = slices

        # 3-d tensor, len * batch_size * n_in
        slices = slices.reshape( (x.shape[0], x.shape[1], n_in) )


        # stacking the feature extraction layers
        depth = args.depth
        self.layers = [ ]
        layers = self.layers
        prev_output = slices
        size = 0
        softmax_inputs = [ ]
        for i in range(depth):
            layer = StrConvLayer(
                      input = prev_output,
                      rng = rng,
                      n_in = n_hidden if prev_output != slices else n_in,
                      n_out = n_hidden,
                      decay = args.decay,
                      order = args.order,
                      activation = args.act
                )
            layers.append(layer)
            prev_output = layer.output
            softmax_inputs.append(T.sum(layer.output, axis=0)) # summing over columns
            size += n_hidden

        # final feature representation is the concatenation of all extraction layers
        softmax_input = T.concatenate(softmax_inputs, axis=1) / x.shape[0]

        # feed the feature repr. to the softmax output layer
        layers.append( SoftmaxLayer(
                input = softmax_input,
                n_in = size,
                n_out = len(vocaby)
        ) )

        # create the same structure with dropout
        if args.dropout_rate > 0:
            dropout_layers = [ ]
            softmax_inputs = [ ]
            dropout_p = args.dropout_rate
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                    rng.randint(9999))

            prev_output = slices
            prev_output = prev_output * srng.binomial(
                n=1, p=1-dropout_p, size=prev_output.shape,
                dtype=theano.config.floatX) / ((1-dropout_p)**0.5)
            for i in range(depth):
                layer = StrConvLayer(
                          input = prev_output,
                          rng = rng,
                          n_in = n_hidden if prev_output != slices else n_in,
                          n_out = n_hidden,
                          decay = args.decay,
                          order = args.order,
                          activation = args.act,
                          P = layers[i].P,
                          Q = layers[i].Q,
                          R = layers[i].R,
                          O = layers[i].O,
                          b = layers[i].b
                    )
                dropout_layers.append(layer)
                prev_output = layer.output
                prev_output = prev_output * srng.binomial(
                    n=1, p=1-dropout_p, size=prev_output.shape,
                    dtype=theano.config.floatX) / (1-dropout_p)
                softmax_inputs.append(T.sum(layer.output, axis=0)) # summing over columns

            softmax_input = T.concatenate(softmax_inputs, axis=1) / x.shape[0]
            softmax_input = softmax_input * srng.binomial(
                n=1, p=1-dropout_p, size=softmax_input.shape,
                dtype=theano.config.floatX) / ((1-dropout_p)**0.5)

            dropout_layers.append( SoftmaxLayer(
                    input = softmax_input,
                    n_in = size,
                    n_out = len(vocaby),
                    W = layers[-1].W
            ) )
        else:
            dropout_layers = layers

        # unnormalized score of y given x
        self.s_y_given_x = dropout_layers[-1].s_y_given_x
        self.p_y_given_x = dropout_layers[-1].p_y_given_x
        self.pred = layers[-1].pred
        self.nll_loss = dropout_layers[-1].nll_loss

        # adding regularizations
        self.l2_sqr = None
        self.params = [ ]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

    def save_model(self, path, args):
         # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args,
                    self.EMB.get_value(), self.vocabx, self.vocaby ),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args, EMB_values, vocabx, vocaby = pickle.load(fin)

        # construct the network and initialize the parameters
        self.args = args
        self.vocabx = vocabx
        self.vocaby = vocaby
        self.ready(EMB_values)
        for x,v in zip(self.params, param_values):
            x.set_value(v)

    # vanilla SGD
    def sgd_update(self, cost, learning_rate):
        gparams = [ T.grad(cost, param) for param in self.params ]
        self.gparams = gparams
        updates = [ ]
        for param, gparam in zip(self.params, gparams):
            if param == self.slices:
                updates.append( (self.EMB, T.inc_subtensor(param, -learning_rate*gparam)) )
            else:
                updates.append( (param, param - learning_rate*gparam) )
        return updates

    def adagrad_update(self, cost, learning_rate, eps=1e-8):
        params = [ p if p != self.slices else self.EMB for p in self.params ]
        accumulators = [ theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                                dtype=theano.config.floatX))
                         for p in params ]
        gparams = [ T.grad(cost, param) for param in self.params ]
        self.gparams = gparams
        updates = [ ]
        for param, gparam, acc in zip(self.params, gparams, accumulators):
            if param == self.slices:
                acc_slices = acc[self.x.flatten()]
                new_acc_slices = acc_slices + gparam**2
                updates.append( (acc, T.set_subtensor(acc_slices, new_acc_slices)) )
                updates.append( (self.EMB, T.inc_subtensor(param,
                                 - learning_rate * gparam / T.sqrt(new_acc_slices+eps))) )
            else:
                new_acc = acc + gparam**2
                updates.append( (acc, new_acc) )
                updates.append( (param, param - learning_rate * gparam /
                                    T.sqrt(new_acc + eps)) )
        return updates


    def create_one_batch(self, ids, x, y):
        batch_x = numpy.column_stack( [ x[i] for i in ids ] )
        batch_y = numpy.array( [ y[i] for i in ids ] )
        return batch_x, batch_y

    # shuffle training examples and create mini-batches
    def create_batches(self, perm, x, y, dropout_p=0, rnd=None):

        if dropout_p > 0:
            dropout_x = [ ]
            for a in x:
                b = [ w for w in a if rnd.random() > dropout_p ]
                if len(b) == 0:
                    b.append( a[random.randint(0, len(a)-1)] )
                dropout_x.append(b)
            x = dropout_x

        # sort sequences based on their length
        # permutation is necessary if we want different batches every epoch
        lst = sorted(perm, key=lambda i: len(x[i]))

        batches_x = [ ]
        batches_y = [ ]
        size = self.args.batch
        ids = [ lst[0] ]
        for i in lst[1:]:
            if len(ids) < size and len(x[i]) == len(x[ids[0]]):
                ids.append(i)
            else:
                bx, by = self.create_one_batch(ids, x, y)
                batches_x.append(bx)
                batches_y.append(by)
                ids = [ i ]
        bx, by = self.create_one_batch(ids, x, y)
        batches_x.append(bx)
        batches_y.append(by)

        # shuffle batches
        batch_perm = range(len(batches_x))
        random.shuffle(batch_perm)
        batches_x = [ batches_x[i] for i in batch_perm ]
        batches_y = [ batches_y[i] for i in batch_perm ]
        return batches_x, batches_y

    def eval_accuracy(self, preds, golds):
        fine = sum([ sum(p == y) for p,y in zip(preds, golds) ]) + 0.0
        fine_tot = sum( [ len(y) for y in golds ] )
        return fine/fine_tot


    def train(self, train, dev, test):
        args = self.args
        trainx, trainy = train

        if dev:
            dev_batches_x, dev_batches_y = self.create_batches(
                    range(len(dev[0])),
                    dev[0],
                    dev[1]
            )

        if test:
            test_batches_x, test_batches_y = self.create_batches(
                    range(len(test[0])),
                    test[0],
                    test[1]
            )

        learning_rate = args.learning_rate

        if args.loss == "nll":
            cost = self.nll_loss(self.y) + self.l2_sqr
        else:
            raise ValueError(
                    "unsupported loss function: {}".format(args.loss)
                )


        if args.learning == "sgd":
            updates = self.sgd_update(cost=cost, learning_rate=learning_rate)
        elif args.learning == "adagrad":
            updates = self.adagrad_update(cost=cost, learning_rate=learning_rate)
        else:
            raise ValueError(
                    "unsupported learning method: {}".format(args.learning)
                )

        gparams = self.gparams
        l2_params = [ param.norm(2) if param != self.slices else self.EMB.norm(2)
                for param in self.params ]

        train_model = theano.function(
             inputs = [self.x, self.y],
             outputs = cost,
             updates = updates,
             allow_input_downcast = True
        )

        eval_l2p = theano.function(
             inputs = [ ],
             outputs = l2_params
        )

        eval_acc = theano.function(
             inputs = [self.x],
             outputs = self.pred,
             allow_input_downcast = True
        )

        unchanged = 0
        best_dev = 0.0
        start_time = time.time()
        eval_period = args.eval_period

        perm = range(len(trainx))

        for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 30: return
            train_loss = 0.0

            random.shuffle(perm)
            batches_x, batches_y = self.create_batches(perm, trainx, trainy)

            N = len(batches_x)
            for i in xrange(N):

                if i % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()

                x = batches_x[i]
                y = batches_y[i]

                va = train_model(x, y)
                train_loss += va

                if (i == N-1) or (eval_period > 0 and (i+1) % eval_period == 0):
                    say( "" )
                    say( "Epoch %.1f\tloss=%.4f\t|p|=%s  [%.2fm]" % (
                            epoch + (i+1)/(N+0.0),
                            train_loss / (i+1),
                            str( [ "%.1f" % x for x in eval_l2p() ] ),
                            (time.time()-start_time) / 60.0
                    ))

                    if dev:
                        preds = [ eval_acc(x) for x in dev_batches_x ]
                        nowf_dev = self.eval_accuracy(preds, dev_batches_y)
                        if nowf_dev > best_dev:
                            unchanged = 0
                            best_dev = nowf_dev
                            if args.model:
                                self.save_model(args.model, args)

                        say("\tdev accuracy=%.4f\tbest=%.4f\n" % (
                                nowf_dev,
                                best_dev
                        ))
                        if args.test and nowf_dev == best_dev:
                            preds = [ eval_acc(x) for x in test_batches_x ]
                            nowf_test = self.eval_accuracy(preds, test_batches_y)
                            say("\ttest accuracy=%.4f\n" % (
                                    nowf_test,
                            ))

                        if best_dev > nowf_dev + 0.05:
                            return

                    start_time = time.time()


def main(args):
    print args

    if args.seed >= 0:
        seed = args.seed
    else:
        seed = int(time.time()*1000) % 9999
    print "seed:", seed
    random.seed(seed)

    model = None
    if args.train:
        vocabx, vocaby = Pipe.create_vocabulary(args)
        if args.embedding:
            vocabx, embeddings = Pipe.load_embeddings(args)
        train_x, train_y = Pipe.read_corpus(args.train, args, vocabx, vocaby)
        if args.dev:
            dev_x, dev_y = Pipe.read_corpus(args.dev, args, vocabx, vocaby)
        if args.test:
            test_x, test_y = Pipe.read_corpus(args.test, args, vocabx, vocaby)
        model = ConvModel(
                  args = args,
                  vocabx = vocabx,
                  vocaby = vocaby
            )
        model.ready( embeddings if args.embedding else None )
        model.train(
                (train_x, train_y),
                (dev_x, dev_y) if args.dev else None,
                (test_x, test_y) if args.test else None,
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data")
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to development data")
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data")
    argparser.add_argument("--cutoff",
            type = int,
            default = 0,
            help = "prune words ocurring <= cutoff")
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 10,
            help = "hidden dimensions")
    argparser.add_argument("--decay",
            type = float,
            default = 0.3)
    argparser.add_argument("--loss",
            type = str,
            default = "nll",
            help = "training loss function")
    argparser.add_argument("--learning",
            type = str,
            default = "adagrad",
            help = "learning method (adagrad, sgd, ...)")
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.01",
            help = "learning rate")
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs")
    argparser.add_argument("--eval_period",
            type = int,
            default = -1,
            help = "evaluate on dev every period")
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0.0,
            help = "dropout probability"
    )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
    )
    #argparser.add_argument("--l2_reg_2",
    #        type = float,
    #        default = 0.0001
    #)
    argparser.add_argument("--embedding",
            type = str,
            default = ""
    )
    argparser.add_argument("--batch",
            type = int,
            default = 15,
            help = "mini-batch size"
    )
    argparser.add_argument("--depth",
            type = int,
            default = 1,
            help = "number of feature extraction layers (min:1)"
    )
    argparser.add_argument("--order",
            type = int,
            default = 3,
            help = "when the order is k, we use up tp k-grams (k=1,2,3)"
    )
    argparser.add_argument("--act",
            type = str,
            default = "relu",
            help = "activation function (none, relu, tanh)"
    )
    argparser.add_argument("--seed",
            type = int,
            default = -1,
            help = "random seed of the model"
    )
    argparser.add_argument("--model",
            type = str,
            default = "",
            help = "save model to this file"
    )
    args = argparser.parse_args()
    main(args)


