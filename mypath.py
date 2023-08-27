class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = './UCF101'

            # Save preprocess data into output_dir
            output_dir = './VAR/ucf101'

            return root_dir, output_dir


    @staticmethod
    def model_dir():
        return './pretrain_model/ucf101-caffe.pth'