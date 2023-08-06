from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource
import os

GB1_DATA_FILE_X = [os.path.join(os.path.dirname(__file__), '..', 'datafiles', f'gb1-x-{shard_i}.npy')
                   for shard_i in range(2)]


class GB1Dataset(DiscreteDataset):

    name = "gb1/substitutions"
    x_name = "sequence"
    y_name = "Fitness"

    @staticmethod
    def register_x_shards():
        return

    @staticmethod
    def register_y_shards():
        return [DiskResource(disk_target=disc_target.replace("-x-", "-y-"), is_absolute=True)
                for disc_target in GB1_DATA_FILE_X]

    def __init__(self, soft_interpolation=0.6, **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        soft_interpolation: float
            floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # initialize the dataset using the method in the base class
        super(GB1Dataset, self).__init__(
            self.register_x_shards(),
            self.register_y_shards(),
            is_logits=False, num_classes=20,
            soft_interpolation=soft_interpolation, **kwargs)