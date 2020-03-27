# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 00:23:22 2020

@author: Matt
"""

from functools import partial
import geopandas as gpd
from networkx import connected_components
import itertools
import numpy as np
from scipy import stats
from returnGerryPlans import getGerryPlans
from gerrychain import (
    Election,
    Graph,
    MarkovChain,
    Partition,
    accept,
    constraints,
    updaters,
)
from gerrychain.proposals import recom, propose_random_flip
from gerrychain.updaters import cut_edges, Tally
from gerrychain.constraints import single_flip_contiguous, contiguous
from gerrychain.accept import always_accept

class NoInitialChain:
    """Stores instance variables of chain excluding initial_state."""
    proposal = None
    constraints = None
    accept = None
    total_steps = None

    def __init__(self, proposal, constraints, accept, total_steps):
        """Constructor with given instance variables."""
        self.proposal = proposal
        self.constraints = constraints
        self.accept = accept
        self.total_steps = total_steps

    def toChain(self, initial_partition):
        """Returns chain with instance variables of self NoInitialChain and
        parameter initial_partition."""
        return MarkovChain(
            proposal=self.proposal,
            constraints=self.constraints,
            accept=self.accept,
            # Declares new Partition with identical instances in order to avoid
            # attempting to access parent
            initial_state=Partition(initial_partition.graph,
                                    assignment=initial_partition.assignment,
                                    updaters=initial_partition.updaters),
            total_steps=self.total_steps
        )

def probDiffClass(dist1, dist2, alpha):
    """
    Returns probability that an outlier from dist1 is classified as an outlier
    according to dist2.

    Parameters:
    -dist1 (list or tuple of numbers): Distrubution 1
    -dist2 (list of tuple of numbers): Distribution 2
    -alpha: 100*alpha and 100*(1-alpha) are the percentile cutoffs of each
    distribution for classifying values as outliers
    """

    if (alpha <= 0 or alpha > 0.5):
        raise ValueError('alpha must be between 0 and 0.5')

    # Cutoff for classifying value as outlier type A according to Distribution 1
    a1 = stats.scoreatpercentile(dist1, 100*alpha)
    # Cutoff for classifying value as outlier type B according to Distribution 1
    b1 = stats.scoreatpercentile(dist1, 100*(1-alpha))
    # Cutoff for classifying value as outlier type A according to Distribution 2
    a2 = stats.scoreatpercentile(dist2, 100*alpha)
    # Cutoff for classifying value as outlier type B according to Distribution 1
    b2 = stats.scoreatpercentile(dist2, 100*(1-alpha))

    gerryCount = 0 # Number of values classified as outlier according to dist1
    misClass = 0 # Number of outliers from dist1 classified differently by dist2
    for val in dist1:
        # If val is an outlier of dist1:
        if (val <= a1 or val >= b1):
            gerryCount += 1
            # If val classified differently by dist2:
            if ((val <= a1 and val > a2) or (val >= b1 and val < b2)):
                misClass += 1

    return misClass / gerryCount # Return probability of misclassification


def outlierDivergence(dist1, dist2, alpha):
    """Defines difference between how distributions classify outliers.

    Choose uniformly from Distribution 1 and Distribution 2, and then choose
    an outlier point according to the induced probability distribution. Returns
    the probability that said point would be classified differently by the other
    distribution.

    Parameters:
    -dist1 (list or tuple of numbers): Distrubution 1
    -dist2 (list of tuple of numbers): Distribution 2
    -alpha: 100*alpha and 100*(1-alpha) are the percentile cutoffs of each
    distribution for classifying values as outliers
    """

    return (probDiffClass(dist1, dist2, alpha) + probDiffClass(dist2, dist1, alpha)) / 2

def allChains(noInitChains, initial_partitions):
    """Returns list of of chains with parameters of noInitChain and initial_state
    of each of initial_partitions."""
    chains = [[] for i in range(len(noInitChains))]
    for i, chain in enumerate(noInitChains):
        for partition in initial_partitions:
            chains[i].append(chain.toChain(partition))

    return chains

def distsMatrix(chains, initial_states, electionName, electionStatistic="seats",
                constraintFunction=(lambda x, y: True)):
    """
    Returns matrix of distributions, where each row corresponds to a chain,
    and each entry corresponds to a distribution produced by the chain of that
    row and the corresponding initial_states.

    Parameters:
    -chains (list of Gerry Markov Chains or NoInitChains): List of chains to be run
    -initial_states (list of partitions): List of partitions to use as intial_states
    for each chain. Defaulted to None, in which case chains must conist of Gerry
    Markov Chains
    -electionStatistic (String): String corresponding to the desired statistic
    to be saved
    -constraintFunction (function): Function used for additional constraint. Must
    take two parameters: the current partition and its index in the chain, in
    that order. Data is only added to distribution if constraintFunction returns
    True. Defaulted to always return True.
    """
    if electionStatistic not in {"seatsDem", "wonDem", "efficiency_gap", "mean_median",
                                 "mean_thirdian", "partisan_bias", "partisan_gini"}:
        raise ValueError('Invalid election statistic')

    # If initial_states is None, then each row is a single distribution
    if initial_states == None:
        for i, chain in enumerate(chains):
            chainMatrix = []
            chainMatrix.append([chain])
            dists = [[] for y in range(len(chains))]
    # Else call allChains to set each row
    else:
        chainMatrix = allChains(chains, initial_states)
        dists = [[[] for x in range(len(initial_states))] for y in range(len(chains))]
    # Set each entry in dists to be the values stored of the corresponding chain
    for i, row in enumerate(chainMatrix):
        for j, chain in enumerate(row):
            dist = []
            for k, partition in enumerate(chain): # Run chain
                # If constraint function is True, add the chosen statistic to
                # distribution:
                if (constraintFunction(partition, k)):
                    if (electionStatistic=="seatsDem" or electionStatistic=="wonDem"):
                        dist.append(partition[electionName].seats("Dem"))
                    elif (electionStatistic=="seatsRep" or electionStatistic=="wonRep"):
                        dist.append(partition[electionName].seats("Rep"))
                    elif (electionStatistic=="efficiency_gap"):
                        dist.append(partition[electionName].efficiency_gap())
                    elif (electionStatistic=="mean_median"):
                        dist.append(partition[electionName].mean_median())
                    elif (electionStatistic=="mean_thirdian"):
                        dist.append(partition[electionName].mean_thirdian())
                    elif (electionStatistic=="partisan_bias"):
                        dist.append(partition[electionName].partisan_bias())
                    elif (electionStatistic=="partisan_gini"):
                        dist.append(partition[electionName].partisan_gini())
                #if (k % 10000 == 0): # May remove: used to keep track of progress
                #    print(i, j, k) # May remove
            dists[i][j] = dist # Add dist to matrix

    return dists

def outlierMatrix(distsMatrix, alphas=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
    """
    Takes a matrix of distributions, and returns the corresponding outlier
    divergence matrix, where the entry at index (i, j) corresponds to the average
    outlier divergence between distributions at index i and those at index j.

    Parameters:
    -distsMatrix (list of list of list of nums): matrix where each row corresponds
    to a set of distributions corresponding to a given chain
    -alphas (list of floats): 100*alpha and 100*(1-alpha) are the percentile
    cutoffs of each distribution for classifying values as outliers. List
    of desired alpha values.
    """
    # List of outlier matrices, each corresponding to an alpha value in order
    matrices = [np.zeros((len(distsMatrix), len(distsMatrix))) for i in range(len(alphas))]
    # Iterate through upper-half triangle of distsMatrix:
    for m in range(len(distsMatrix)):
        for n in range(m, len(distsMatrix)):
            # Set of pairs of distributions, choosing one corresponding to
            # chain m and the other from chain n
            pairs = list(itertools.product(distsMatrix[m], distsMatrix[n]))
            # Finds the average Outlier Divergence over all such pairs for the
            # appropriate alpha
            for j, alpha in enumerate(alphas):
                divergence = 0.0
                for pair in pairs:
                    divergence += outlierDivergence(pair[0], pair[1], alpha)
                    averageDiv = divergence / len(pairs)
                    # Add value to corresponding entries in matrix:
                    matrices[j][m][n] = averageDiv
                    matrices[j][n][m] = averageDiv
    return matrices


def printOutlierMatrices(matrices, alphas=[0.1, 0.05, 0.01, 0.005, 0.001,
                                           0.0005, 0.0001]):
    """Prints matrices in desired format."""
    for matrix, alpha in zip(matrices, alphas):
        print("alpha:", alpha, "\n")
        print(matrix)
        print("\n")

# Import graph and prepare to run chain:

graph = Graph.from_file("./Data/Wisconsin/WI_ltsb_corrected_final.shp")

islands = graph.islands
components = list(connected_components(graph))
df = gpd.read_file("./Data/Wisconsin/WI_ltsb_corrected_final.shp")
df.to_crs({"init": "epsg:26986"}, inplace=True)


biggest_component_size = max(len(c) for c in components)
problem_components = [c for c in components if len(c) != biggest_component_size]
problem_nodes = [node for component in problem_components for node in component]
problem_geoids = [graph.nodes[node]["GEOID10"] for node in problem_nodes]

largest_component_size = max(len(c) for c in components)
to_delete = [c for c in components if len(c) != largest_component_size]
for c in to_delete:
    for node in c:
        graph.remove_node(node)


election = Election("PRETOT16", {"Dem": "PREDEM16", "Rep": "PREREP16"})

#Create initial parition based on congressional districts
initial_partition = Partition(
    graph,
    assignment="CON",
    updaters={
        "cut_edges": cut_edges,
        "population": Tally("PERSONS", alias="population"),
        "PRETOT16": election
    }
)

# Example set of NoInitialChains to run:

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.06)
compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

chainFlipAlwaysShort = NoInitialChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous,
                 pop_constraint,
                 compactness_bound],
    accept=always_accept,
    total_steps=75000
)

chainFlipAlwaysLong = NoInitialChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous,
                 pop_constraint,
                 compactness_bound],
    accept=always_accept,
    total_steps=500000
)

my_updaters = {"population": Tally("PERSONS", alias="population"), "PRETOT16": election}
ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col="PERSONS",
                   pop_target=ideal_population,
                   epsilon=0.06,
                   node_repeats=2
                  )


chainRecomShort = NoInitialChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound,
        contiguous
    ],
    accept=accept.always_accept,
    total_steps=50000
)

chainRecomLong = NoInitialChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound,
        contiguous
    ],
    accept=accept.always_accept,
    total_steps=75000
)

testChains = [chainFlipAlwaysShort, chainFlipAlwaysLong, chainRecomShort, chainRecomLong]
testAlphas=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
initial_partitions = getGerryPlans(75, 0.15)
print("Number of starting partitions:", len(initial_partitions))
distsMatrix = distsMatrix(chains=testChains, initial_states=initial_partitions,
                          electionName="PRETOT16", electionStatistic="wonDem")
matrix = outlierMatrix(distsMatrix, alphas=testAlphas)
printOutlierMatrices(matrix, testAlphas)






