import os,sys,itertools
import theano, numpy
import theano.tensor as T
from model import ConvModel, Pipe

numpy.set_printoptions(precision=3)

def run_demo(model):
    args = model.args
    vocabx, vocaby = model.vocabx, model.vocaby
    print "vocab size: {}".format(len(vocabx))
    id2word = dict( [ (y,x) for x,y in vocabx.items() ] )
    id2label = dict( [ (y,x) for x,y in vocaby.items() ] )
    print id2label

    score_vec = numpy.array( [ float(id2label[x])-2.0 for x in range(5) ] )

    outputs = [ ]
    for layer in model.layers[:-1]:
        outputs.append( layer.output.reshape(
                            (layer.output.shape[0],
                            layer.output.shape[2])
                        ) )  # len * 1 * h --> len * h
    softmax_input = T.concatenate(outputs, axis=1)  # len * (h*depth)
    s_y_given_x = T.dot(softmax_input, model.layers[-1].W)
    p_y_given_x = T.nnet.softmax(s_y_given_x)  # len * n_label

    g = theano.function(
            inputs = [ model.x ],
            outputs = T.sum(p_y_given_x * score_vec, axis=1)
        )

    predict = theano.function(
            inputs = [ model.x ],
            outputs = model.pred
        )

    print ""

    while True:
        sys.stdout.write("input> ")
        sys.stdout.flush()
        line = sys.stdin.readline().strip().lower()
        if line == "quit": break

        x,y = Pipe.process_line("0 "+line)
        x = Pipe.map_to_idx(x, vocabx, args)
        x = x.reshape(len(x), 1)

        scores = g(x)
        pred = predict(x)[0]
        sys.stdout.write("\n")
        sys.stdout.write("Sentiment:\t{}\n".format(id2label[pred]))
        sys.stdout.write("Sentiment at each position:\t"+"\t".join([ "%.2f" % x for x in scores]))
        sys.stdout.write("\n\n")
        sys.stdout.flush()

def main():
    model = ConvModel(
            args = None,
            vocabx = None,
            vocaby = None
        )
    model_path = sys.argv[1]
    model.load_model(model_path)

    print model.args
    run_demo(model)

if __name__ == "__main__":
    main()

