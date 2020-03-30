
import numpy as np
import math
import warnings
import datetime
import optparse
import os, errno
import matplotlib.pyplot as plt
import pandas as pd

def plot_fig_image(history,epochs, outfile):
    fig = plt.figure()
    plt.plot(range(1,epochs+1),history['image_output_acc'],label='training-acc')
    plt.plot(range(1,epochs+1),history['image_output_loss'],label='training-loss')
    plt.plot(range(1,epochs+1),history['val_image_output_acc'],label='validation-acc')
    plt.plot(range(1,epochs+1),history['val_image_output_loss'],label='validation-loss')

    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy-loss')
    plt.xlim([1,epochs])
#     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Performance")
    # plt.show()
    fig.savefig(outfile)
    plt.close(fig)

def plot_fig_text(history,epochs, outfile):
    fig = plt.figure()
    plt.plot(range(1,epochs+1),history['acc'],label='training-acc')
    plt.plot(range(1,epochs+1),history['loss'],label='training-loss')
    plt.plot(range(1,epochs+1),history['val_acc'],label='validation-acc')
    plt.plot(range(1,epochs+1),history['val_loss'],label='validation-loss')

    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy-loss')
    plt.xlim([1,epochs])
#     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Performance")
    # plt.show()
    fig.savefig(outfile)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="data_file")
    parser.add_option('-o', action="store", dest="out_file")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    data_file = options.data_file
    out_file = options.out_file
    out_dir = os.path.dirname(out_file)
    out_base =os.path.basename(out_file)
    out_base = os.path.splitext(out_base)[0]
    # out_file_image = out_dir + "/" + out_base +"_image.jpg"
    out_file_text = out_dir + "/" + out_base + "_text.jpg"

    # train_img_file = options.data_file  # "exp/iraq_earthquake_task_text_train.csv"

    df = pd.read_csv(data_file, sep="\t")
    epochs=df.shape[0]
    # print(df)

    # plot_fig_image(df,epochs, out_file_image)
    plot_fig_text(df, epochs, out_file_text)
