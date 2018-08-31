import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.l1=L.Linear(None, 300, initialW=initialW)
            self.l2=L.Linear(300, 300, initialW=initialW)
            self.l3=L.Linear(300, 2, initialW=initialW)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.dropout(F.relu(self.l2(h)),ratio=.2)
        h = self.l3(h)

        return h

class MLP2(chainer.Chain):
    def __init__(self):
        super(MLP2, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.l1=L.Linear(None, 1024, initialW=initialW)
            self.l2=L.Linear(1024, 2048, initialW=initialW)
            self.l3=L.Linear(2048, 1024, initialW=initialW)
            self.l4=L.Linear(1024, 512, initialW=initialW)
            self.l5=L.Linear(512, 256, initialW=initialW)
            self.l6=L.Linear(256, 2, initialW=initialW)
            #self.bn1 = L.BatchNormalization(1024)

    def __call__(self, x):
        #h = self.bn1(F.relu(self.l1(x)))
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.dropout(F.relu(self.l3(h)))
        h = F.relu(self.l4(h))
        h = F.dropout(F.relu(self.l5(h)))
        h = self.l6(h)

        return h
       
