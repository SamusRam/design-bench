from design_bench.oracles.approximate_oracle import ApproximateOracle
from design_bench.datasets.dataset_builder import DatasetBuilder
from design_bench.disk_resource import DiskResource
import abc


class RandomForestOracle(OracleBuilder):
    """An abstract class for managing the ground truth score functions f(x)
    for model-based optimization problems, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    dataset: DatasetBuilder
        an instance of a subclass of the DatasetBuilder class which has
        a set of design values 'x' and prediction values 'y', and defines
        batching and sampling methods for those attributes

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

    score(np.ndarray) -> np.ndarray:
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

    @abc.abstractmethod
    def is_downloaded(self, dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        file_name = f"RandomForestOracle/{type(dataset).__name__}.zip"
        resource = DiskResource(
            file_name, is_absolute=False, download_method="direct",
            download_target=f"https://design-bench."
                            f"s3-us-west-1.amazonaws.com/{file_name}")

        return resource.is_downloaded

    @abc.abstractmethod
    def download(self, dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    @abc.abstractmethod
    def can_download(self, dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model, dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def fit(dataset, **kwargs):
        """a function that accepts a set of design values 'x' and prediction
        values 'y' and fits an approximate oracle to serve as the ground
        truth function f(x) in a model-based optimization problem

        Arguments:

        dataset: DatasetBuilder
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes

        """

        raise NotImplementedError

    def protected_score(self, x):
        """Score function to be implemented by oracle subclasses, where x is
        either a batch of designs if self.is_batched is True or is a
        single design when self._is_batched is False

        Arguments:

        x_batch: np.ndarray
            a batch or single design 'x' that will be given as input to the
            oracle model in order to obtain a prediction value 'y' for
            each 'x' which is then returned

        Returns:

        y_batch: np.ndarray
            a batch or single prediction 'y' made by the oracle model,
            corresponding to the ground truth score for each design
            value 'x' in a model-based optimization problem

        """

        return self.model.predict(x)
