from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.datasets.discrete.pyrosetta_energy_dataset import PyrosettaEnergyDataset
import numpy as np
import pyrosetta as prs

aa_single_to_three_letter_code = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "B": "ASX",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "Z": "GLX",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


class PyrosettaEnergyDummyOracle(ExactOracle):
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

    """

    name = "pyrosetta_energy_dummy"

    @classmethod
    def supported_datasets(cls):
        """An attribute the defines the set of dataset classes which this
        oracle can be applied to forming a valid ground truth score
        function for a model-based optimization problem

        """

        return {PyrosettaEnergyDataset}

    @classmethod
    def fully_characterized(cls):
        """An attribute the defines whether all possible inputs to the
        model-based optimization problem have been evaluated and
        are are returned via lookup in self.predict

        """

        return False

    @classmethod
    def is_simulated(cls):
        """An attribute the defines whether the values returned by the oracle
         were obtained by running a computer simulation rather than
         performing physical experiments with real data

        """

        return True

    def __init__(self, dataset: DiscreteDataset, **kwargs):
        """Initialize the ground truth score function f(x) for a model-based
        optimization problem, which involves loading the parameters of an
        oracle model and estimating its computational cost

        Arguments:

        dataset: DiscreteDataset
            an instance of a subclass of the DatasetBuilder class which has
            a set of design values 'x' and prediction values 'y', and defines
            batching and sampling methods for those attributes
        pdb_path

        """
        prs.init("-mute all")

        # We will reuse this pose over and over, mutating it to match
        # whatever sequence we are given to measure.
        # This is necessary since sequence identity can only be mutated
        # one residue at a time in Rosetta, because the atom coords of the
        # backbone of the previous residue are copied into the new one.
        self.pose = prs.pose_from_pdb(pdb_file)
        self.wt_pose = self.pose.clone()

        # Change self.pose from full-atom to centroid representation
        to_centroid_mover = prs.SwitchResidueTypeSetMover("centroid")
        to_centroid_mover.apply(self.pose)

        # Use 1 - sigmoid(centroid energy / norm_value) as the fitness score
        self.score_function = prs.create_score_function("cen_std")
        self.sigmoid_center = sigmoid_center
        self.sigmoid_norm_value = sigmoid_norm_value

        # initialize the oracle using the super class
        super(PyrosettaEnergyDummyOracle, self).__init__(
            dataset, internal_batch_size=1, is_batched=False,
            expect_normalized_y=False,
            expect_normalized_x=False, expect_logits=False, **kwargs)

    def _mutate_pose(self, mut_aa: str, mut_pos: int):
        """Mutate `self.pose` to contain `mut_aa` at `mut_pos`."""
        current_residue = self.pose.residue(
            mut_pos + 1
        )  # + 1 since rosetta is 1-indexed
        conformation = self.pose.conformation()

        # Get ResidueType for new residue
        new_restype = prs.rosetta.core.pose.get_restype_for_pose(
            self.pose, aa_single_to_three_letter_code[mut_aa]
        )

        # Create the new residue using current_residue backbone
        new_res = prs.rosetta.core.conformation.ResidueFactory.create_residue(
            new_restype,
            current_residue,
            conformation,
            preserve_c_beta=False,
            allow_alternate_backbone_matching=True,
        )

        # Make sure we retain as much info from the previous resdiue as possible
        prs.rosetta.core.conformation.copy_residue_coordinates_and_rebuild_missing_atoms(  # noqa: E501
            current_residue,
            new_res,
            conformation,
            preserve_only_sidechain_dihedrals=False,
        )

        # Replace residue
        self.pose.replace_residue(mut_pos + 1, new_res, orient_backbone=False)

        # Update the coordinates of atoms that depend on polymer bonds
        conformation.rebuild_polymer_bond_dependent_atoms_this_residue_only(mut_pos + 1)

    def get_folding_energy(self, sequence: str):
        """
        Return rosetta folding energy of the given sequence in
        `self.pose`'s conformation.
        """
        pose_sequence = self.pose.sequence()

        if len(sequence) != len(pose_sequence):
            raise ValueError(
                "`sequence` must be of the same length as original protein in .pdb file"
            )

        # Mutate `self.pose` where necessary to have the same sequence identity as
        # `sequence`
        for i, aa in enumerate(sequence):
            if aa != pose_sequence[i]:
                self._mutate_pose(aa, i)

        return self.score_function(self.pose)

    def protected_predict(self, x):
        """
        Taken from FLEXS

        Negate and normalize folding energy to get maximization objective

        Score function to be implemented by oracle subclasses, where x is
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
        energies = torch.tensor([self.get_folding_energy(seq) for seq in x])
        scaled_energies = (-energies - self.sigmoid_center) / self.sigmoid_norm_value
        return torch.sigmoid(scaled_energies).numpy()


        # raise NotImplementedError("For the Dummy Pyrosetta Oracle the code should not access oracle's predict function")
