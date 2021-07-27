import os


class DatasetCatalog:
    DATA_DIR = 'E:\Datasets\VOCdevkit'
    DATASETS = {
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'dt_voc_2007_train': {
            "data_dir": "dt_clipart/VOC2007",
            "split": "train"
        },
        'dt_voc_2007_val': {
            "data_dir": "dt_clipart/VOC2007",
            "split": "val"
        },
        'dt_voc_2007_trainval': {
            "data_dir": "dt_clipart/VOC2007",
            "split": "trainval"
        },
        'dt_voc_2007_test': {
            "data_dir": "dt_clipart/VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'dt_voc_2012_train': {
            "data_dir": "dt_clipart/VOC2012",
            "split": "train"
        },
        'dt_voc_2012_val': {
            "data_dir": "dt_clipart/VOC2012",
            "split": "val"
        },
        'dt_voc_2012_trainval': {
            "data_dir": "dt_clipart/VOC2012",
            "split": "trainval"
        },
        'dt_voc_2012_test': {
            "data_dir": "dt_clipart/VOC2012",
            "split": "test"
        },
        'clipart1k_train': {
            "data_dir": "clipart",
            "split": "train"
        },
        'clipart1k_test': {
            "data_dir": "clipart",
            "split": "test"
        },
    }

    @staticmethod
    def get(name):
        voc_root = DatasetCatalog.DATA_DIR
        if 'VOC_ROOT' in os.environ:
            voc_root = os.environ['VOC_ROOT']

        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            data_dir=os.path.join(voc_root, attrs["data_dir"]),
            split=attrs["split"],
        )
        return dict(factory="VOCDataset", args=args)

