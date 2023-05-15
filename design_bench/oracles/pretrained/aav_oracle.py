from design_bench.oracles.approximate_oracle import ApproximateOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
import tempfile
import math
import numpy as np
import shutil


class AvvCnnOracle(ApproximateOracle):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    external_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which points to
        the mutable task dataset for a model-based optimization problem

    internal_dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has frozen
        statistics and is used for training the oracle

    is_batched: bool
        a boolean variable that indicates whether the evaluation function
        implemented for a particular oracle is batched, which effects
        the scaling coefficient of its computational cost

    internal_batch_size: int
        an integer representing the number of design values to process
        internally at the same time, if None defaults to the entire
        tensor given to the self.score method
    internal_measurements: int
        an integer representing the number of independent measurements of
        the prediction made by the oracle, which are subsequently
        averaged, and is useful when the oracle is stochastic

    noise_std: float
        the standard deviation of gaussian noise added to the prediction
        values 'y' coming out of the ground truth score function f(x)
        in order to make the optimization problem difficult

    expect_normalized_y: bool
        a boolean indicator that specifies whether the inputs to the oracle
        score function are expected to be normalized
    expect_normalized_x: bool
        a boolean indicator that specifies whether the outputs of the oracle
        score function are expected to be normalized
    expect_logits: bool
        a boolean that specifies whether the oracle score function is
        expecting logits when the dataset is discrete

    Public Methods:

    predict(np.ndarray) -> np.ndarray:
        a function that accepts a batch of design values 'x' as input and for
        each design computes a prediction value 'y' which corresponds
        to the score in a model-based optimization problem

    check_input_format(DatasetBuilder) -> bool:
        a function that accepts a list of integers as input and returns true
        when design values 'x' with the shape specified by that list are
        compatible with this class of approximate oracle

    fit(np.ndarray, np.ndarray):
        a function that accepts a data set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

    """

    name = "avv_cnn"
    default_model_kwargs = dict(hidden_size=256, feed_forward_size=256,
                                activation='relu', num_heads=8,
                                num_blocks=4, epochs=20,
                                shuffle_buffer=5000, learning_rate=0.0001,
                                warm_up_steps=4000, dropout_rate=0.1)

    def __init__(self, dataset, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        # initialize the oracle using the super class
        super(AvvCnnOracle, self).__init__(
            dataset, is_batched=True, internal_measurements=1,
            expect_normalized_y=True,
            expect_normalized_x=not isinstance(dataset, DiscreteDataset),
            expect_logits=False if isinstance(
                dataset, DiscreteDataset) else None, **kwargs)

    def save_model_to_zip(self, model, zip_archive):
        pass

    def load_model_from_zip(self, zip_archive):
        pass

    def protected_fit(self, training, validation, model_kwargs=None):
        pass

    def check_input_format(cls, dataset):
        return True

    def protected_predict(self, x, model=None):
        return [-1]
