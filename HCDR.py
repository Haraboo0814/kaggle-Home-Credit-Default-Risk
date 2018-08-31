import argparse
import chainer
import chainer.links as L
import chainer.functions as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from models import MLP
import dataset_loader as dl
import csv


def main():
    parse = argparse.ArgumentParser(description="HCDR")
    parse.add_argument("--batchsize","-b",type=int, default=1000,help="Number if images in each mini batch")
    parse.add_argument("--epoch","-e",type=int, default=10,help="Number of sweeps over the dataset to train")
    parse.add_argument("--gpu","-g",type=int, default=0,help="GPU ID(negative value indicates CPU")
    args = parse.parse_args()

    print("GPU: {}".format(args.gpu))
    print("# Minibatch-size: {}".format(args.batchsize))
    print("# epoch: {}".format(args.epoch))
    print("")

    model = MLP.MLP()

    
    #GPU使用の有無
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    xp = chainer.cuda.cupy if args.gpu >= 0 else np
    
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    #データセットの読み込み
    data_dir = "C:/Users/Kyohei Harada/Desktop/HomeCreditDefaultRisk/all"
    loader = dl.DatasetLoader(data_dir, undersample=True, pca=False)
    train, valid = loader.load_train_test()
    test = loader.load_predict_data()

    #データ数
    train_datanum = train.length
    valid_datanum = valid.length
    print("train:{} valid:{}".format(train_datanum, valid_datanum))

    #プロット用リスト
    plot_epoch = []
    plot_train = []
    plot_valid = []

    print("start learning______________________________")

    for epoch in range(1,args.epoch+1):
        #学習
        perm = np.random.permutation(train_datanum)#0~datanumの数のランダムな並び替え
        plot_epoch.append(epoch)
        sum_loss = 0
        #sum_prec = 0
        sum_acc = 0
        sum_auc = 0
        for i in range(0,train_datanum,args.batchsize):
            x = xp.asarray(train.x[perm[i:i + args.batchsize]]).astype(xp.float32)
            y = xp.asarray(train.y[perm[i:i + args.batchsize]]).astype(xp.int32)
            x_batch = chainer.Variable(x)
            y_batch = chainer.Variable(y)
            if epoch == 1 and i == 0:
                print(x_batch.shape)
                print(y_batch.shape)
                
            model.zerograds()#勾配の削除
            out = model(x_batch)#lossの計算
            loss = F.softmax_cross_entropy(out,y_batch)
            acc = F.accuracy(out,y_batch)
            fpr, tpr, _ = roc_curve(chainer.cuda.to_cpu(y), xp.asnumpy(out.data[:,1]),pos_label=1)
            auc_score = auc(fpr,tpr)
            loss.backward()#lossを伝搬
            optimizer.update()#optimizerの更新
            sum_loss += loss.data * len(x_batch)#loss計算用
            sum_acc += acc.data * len(x_batch)
            sum_auc += auc_score * len(x_batch)

        average_loss = sum_loss / train_datanum
        average_acc = sum_acc / train_datanum
        average_auc = sum_auc / train_datanum

        print ("")
        print ("train:  epoch: {} ,loss: {} ,acc: {} ,auc score: {}".format( epoch, average_loss, average_acc, average_auc))
        plot_train.append(average_loss)

        #検証
        sum_loss = 0
        sum_acc = 0
        #sum_prec = 0
        sum_auc = 0
        for i in range(0,valid_datanum,args.batchsize):
            #データ型をchainer用に変更
            x = xp.asarray(valid.x[i:i + args.batchsize]).astype(xp.float32)
            y = xp.asarray(valid.y[i:i + args.batchsize]).astype(xp.int32)
            x_batch = chainer.Variable(x)
            y_batch = chainer.Variable(y)
            out = model(x_batch)#lossの計算
            loss = F.softmax_cross_entropy(out,y_batch)
            acc = F.accuracy(out,y_batch)
            fpr, tpr, _ = roc_curve(chainer.cuda.to_cpu(y), xp.asnumpy(out.data[:,1]),pos_label=1)
            auc_score = auc(fpr,tpr)
            sum_loss += loss.data * len(x_batch)#loss計算用
            sum_acc += acc.data * len(x_batch)
            sum_auc += auc_score * len(x_batch)
                    
        average_loss = sum_loss / valid_datanum
        average_acc = sum_acc / train_datanum
        average_auc = sum_auc / valid_datanum

        print ("valid:  epoch: {} ,loss: {} ,acc: {} ,auc acore: {}".format( epoch, average_loss, average_acc, average_auc))
        plot_valid.append(average_loss)
    

    #学習-検証のlossを保存
    plt.plot(plot_epoch, plot_train, color="blue", label="train")
    plt.plot(plot_epoch, plot_valid, color="orange", label="valid")
    plt.savefig("result/loss.png")
    plt.close()


    model.to_cpu()
    x = np.array(test.testx).astype(np.float32)
    test_id = np.array(test.testid).astype(np.int)
    with open("predict.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        for i in range(x.shape[0]):
            #writer.writerow(F.softmax(model(x[i].reshape([1,224]))).data[0])
            writer.writerow(F.softmax(model(x[i].reshape([1,444]))).data[0])
            #writer.writerow(F.softmax(model(x[i].reshape([1,100]))).data[0])

if __name__ == "__main__":
    main()