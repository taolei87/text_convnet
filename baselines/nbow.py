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
        # prune punctuations and single characters
        #x = [ w for w in x if len(w.decode("utf8")) > 1 ]
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

class NBoW_Model:
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
        self.layers = [ ]
        layers = self.layers

        softmax_input = T.sum(slices, axis=0) / x.shape[0]
        size = n_in

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
            prev_output = prev_output * srng.binomial(n=1,
                p=1-dropout_p, size=prev_output.shape) / ((1-dropout_p)**0.5)

            softmax_input = T.sum(prev_output, axis=0) / x.shape[0]

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
        l2_gparams = [ theano.shared(value=numpy.asarray(0.0, dtype=theano.config.floatX))
                for param in self.params  ]
        max_gparams = [ theano.shared(value=numpy.asarray(0.0, dtype=theano.config.floatX))
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

                if (i == N-1) or ((i+1) % eval_period == 0):
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
        model = NBoW_Model(
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
            default = 200,
            help = "maximum # of epochs")
    argparser.add_argument("--eval_period",
            type = int,
            default = 10000,
            help = "evaluate on dev every period")
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0.0,
            help = "dropout probability"
    )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.0001
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
    args = argparser.parse_args()
    main(args)


