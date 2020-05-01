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

    if (alpha < 0 or alpha > 0.5):
        raise ValueError('alpha must be between 0 and 0.5')

    # Note that percentile is determined according to scipy.stat's default
    # of fractional interpolation:
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

def distsMatrices(chains, initial_states, electionName, electionStatistics=["efficiency_gap"],
                  party=None, constraintFunction=(lambda x, y: True)):
    """
    Returns a list of lists of lists, where each outer lists corresponds to a
    election statistic, each middle list corresponds to a set of chain parameters, 
    and each inner list corresponds to an initial state and is the distribution 
    generated by its corresponding chain and initial state while saving its 
    corresponding initial partition.

    Parameters:
    -chains (list of Gerry Markov Chains or NoInitChains): List of chains to be run
    -initial_states (list of Partitions): List of partitions to use as intial_states
    for each chain. Defaulted to None, in which case chains must conist of Gerry
    Markov Chains
    -electionStatistic (list of Strings): each String corresponds to a desired 
    statistic to for a matrix
    -party (String): Party to which electoral statistic is with respect to (only 
    applicable to "seats" and "wins"- else party is determined by first party 
    listed in Election for chain)
    -constraintFunction (function): Function used for additional constraint. Must
    take two parameters: the current partition and its index in the chain, in
    that order. Data is only added to distribution if constraintFunction returns
    True. Defaulted to always return True.
    """
    for electionStatistic in electionStatistics:
        if electionStatistic not in {"seats", "won", "efficiency_gap", "mean_median",
                                     "mean_thirdian", "partisan_bias", "partisan_gini"}:
            raise ValueError('Invalid election statistic: ' + electionStatistic)

    # If initial_states is None, then each row is a single distribution
    if initial_states == None:
        for i, chain in enumerate(chains):
            chainMatrix = []
            chainMatrix.append([chain])
            dists = [[[] for y in range(len(chains))] for z in range(len(electionStatistics))]
    # Else call allChains to set each row
    else:
        chainMatrix = allChains(chains, initial_states)
        dists = [[[[] for x in range(len(initial_states))]
                  for y in range(len(chains))] for z in range(len(electionStatistics))]
    for n, electionStatistic in enumerate(electionStatistics):
            # Else call allChains to set each row
            # Set each entry in dists to be the values stored of the corresponding chain
            for i, row in enumerate(chainMatrix):
                for j, chain in enumerate(row):
                    dist = []
                    for k, partition in enumerate(chain): # Run chain
                        # If constraint function is True, add the chosen statistic to
                        # distribution:
                        if (constraintFunction(partition, k)):
                            if (electionStatistic=="seats" or electionStatistic=="won"):
                                dist.append(partition[electionName].seats(party))
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
                    dists[n][i][j] = dist # Add dist to matrix

    return dists

def distsToOutlierMatrices(distsMatrices, alphas=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
    """
    Takes a list of lists of distributions, and returns the corresponding list of 
    lists of outlier divergence matrices, where each sublist corresponds to an
    election statistic, each matrix corresponds a an alpha value, and the entry at index (i, j)
    of a given matrix corresponds to the average outlier divergence between 
    chains i and j of the corresponding election statistic in distMatrices.

    Parameters:
    -distsMatrix (list of list of list of nums): each outer lists corresponds to a
    election statistic, each middle list corresponds to a set of chain parameters, 
    and each inner list corresponds to an initial state and is the distribution 
    generated by its corresponding chain and initial state while saving its 
    corresponding initial partition.
    -alphas (list of floats): 100*alpha and 100*(1-alpha) are the percentile
    cutoffs of each distribution for classifying values as outliers. List
    of desired alpha values.
    """
    # List of lists of outlier matrices, with the outer list corresponding to
    # the election statistics from distMatrices, and the inner list corresponding
    # to the chosen alphas.
    matrices = [[np.zeros((len(distsMatrices[i]), len(distsMatrices[i])))
                 for j in range(len(alphas))] for i in range(len(distsMatrices))]

    # Each dist matrix corresponds to a statistic
    for i, distsMatrix in enumerate(distsMatrices):
            # Iterate through upper-half triangle of distsMatrix:
            for m in range(len(distsMatrix)):
                for n in range(m, len(distsMatrix)):
                    # Get pairs of distributions from chosen chains
                    if n != m:
                        # Set of pairs of distributions, choosing one corresponding to
                        # chain m and the other from chain n
                        pairs = list(itertools.product(distsMatrix[m], distsMatrix[n]))
                    else:
                        # Set pairs of distribution, consisting of all 2-combinations
                        # without replacement
                        pairs = list(itertools.combinations(distsMatrix[m], 2))
                    # Finds the average Outlier Divergence over all such pairs for the
                    # appropriate alpha
                    for j, alpha in enumerate(alphas):
                        divergence = 0.0
                        for pair in pairs:
                            divergence += outlierDivergence(pair[0], pair[1], alpha)
                            averageDiv = divergence / len(pairs)
                            # Add value to corresponding entries in matrix:
                            matrices[i][j][m,n] = averageDiv
                            matrices[i][j][n,m] = averageDiv
    return matrices

def averageOutlierMatrices(outlierList):
    """
    Given a list of lists of lists of outlier matrices, returns a list of lists
    the corresponding averages of the outlier matrices, assuming each entry in
    the outer list is a list of outlier matrices generated by the same set of 
    parameters

    Note that a similar effect is achieved by simply calling
    distsToOutlierMatrices() on a set of distribution matrices with repeated
    initial partitions. Doing so would be more computationally efficient in the
    sense that if there are n initial_partitions then each entry in each matrix
    is the average of the Outlier Discrepency of either n**2 or n*(n-1) (depending
    upon whether the entry is diagonal or not) chains. Furthermore, by the
    linearity of expectation, each entry is an unbiased estimator of the expected
    Outlier Discrepency between chains with the corresponding parameters. However,
    such pairs are not sampled independently, limiting the ability to apply 
    rigorous statistics. Generating a list of out
    """
    # List of list of matrices, with the outer list corresponding to the chosen
    # electoral statistics, the inner list corresponding to the chosen alphas,
    # and each matrix the size of the original outlier matrices
    matrices = [[np.zeros(np.shape(outlierList[0][i][j])) for j in 
                 range(len(outlierList[0][i]))] for i in range(len(outlierList[0]))]
    
    for j in range(len(outlierList[0])):
        for k in range(len(outlierList[0][j])):
            for i in range((len(outlierList))):
                matrices[j][k] = matrices[j][k] + outlierList[i][j][k]
            matrices[j][k] /= len(outlierList)

    return matrices

def printOutlierMatrices(matrices, alphas=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005,
                                           0.0001], electionStatistics=["efficiency_gap"]):
    """Prints matrices in desired format."""
    for i, electionStatistic in enumerate(electionStatistics):
        print("\n\n", electionStatistic, "\n")
        for matrix, alpha in zip(matrices[i], alphas):
            print("alpha:", alpha, "\n")
            print(matrix)
            print("\n")

if __name__ == '__main__':
    """Example usage of code"""
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
        total_steps=50
        )

    chainFlipAlwaysLong = NoInitialChain(
        proposal=propose_random_flip,
        constraints=[single_flip_contiguous,
                     pop_constraint,
                     compactness_bound],
        accept=always_accept,
        total_steps=150
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
        total_steps=10
        )

    chainRecomLong = NoInitialChain(
        proposal=proposal,
        constraints=[
            pop_constraint,
            compactness_bound,
            contiguous
            ],
        accept=accept.always_accept,
        total_steps=20
        )

    testChains = [chainFlipAlwaysShort, chainFlipAlwaysLong, chainRecomShort, chainRecomLong]
    testAlphas=[0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    initial_partitions = getGerryPlans(5, 0.15, electionName="PRETOT16")
    print("Number of starting partitions:", len(initial_partitions))
    testDistMatrices = []
    for i in range(3):
        testDistMatrices.append(distsToOutlierMatrices(distsMatrices(chains=testChains, initial_states=initial_partitions,
                                electionName="PRETOT16",
                                electionStatistics=["won","efficiency_gap"], party="Dem")))

    printOutlierMatrices(averageOutlierMatrices(testDistMatrices), electionStatistics=["won","efficiency_gap"])








