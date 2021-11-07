from glob import glob

import numpy as np
import yaml

if __name__ == '__main__':

    log_dir = './runs/repeat/*'
    for mdl_dir in glob(log_dir):
        mdl = mdl_dir.split('/')[-1]
        err1 = []
        err5 = []
        for datetime_dir in glob(mdl_dir + '/*'):
            datetime = datetime_dir.split('/')[-1]
            for filename_dir in glob(datetime_dir + '/*'):
                filename = filename_dir.split('/')[-1]

                if filename == 'result.yml':
                    # print(dir1)
                    result = yaml.unsafe_load(open(filename_dir))
                    top1_acc = result['top1_acc']
                    top5_acc = result['top5_acc']
                    if mdl == 'svm':
                        top1_acc *= 100.
                        top5_acc *= 100.
                    top1_err = 100. - top1_acc
                    top5_err = 100. - top5_acc

                    err1.append(top1_err)
                    err5.append(top5_err)

        err1 = np.array(err1)
        err5 = np.array(err5)

        print('{}: err1: {}({}), err5: {}({}), num_repeat: {}'.format(mdl,
                                                                      np.round(err1.mean(), 2),
                                                                      np.round(err1.std(), 2),
                                                                      np.round(err5.mean(), 2),
                                                                      np.round(err5.std(), 2),
                                                                      err1.shape[0]))
