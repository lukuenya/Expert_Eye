# imports
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from scipy import stats

from random import choices
import numpy as np
from math import floor
from random import sample

# Decision Tree Classifier

class Node(object):
    """
    Class to define & control tree nodes
    """
    def __init__(self) -> None:
        """
        Initializer for a Node class instance
        """
        self.__split = None
        self.__feature = None
        self.__left = None
        self.__right = None
        self.leaf_value = None

    def set_params(self, split: float, feature: int) -> None:
        """
        Set the split & feature parameters for this node

        Input:
            split   -> value to split feature on
            feature -> index of feature to be used in splitting 
        """
        self.__split = split
        self.__feature = feature

    def get_params(self) -> Tuple[float, int]:
        """
        Get the split & feature parameters for this node

        Output:
            Tuple containing (split,feature) pair
        """
        return (self.__split, self.__feature)

    def set_children(self, left: Node, right: Node) -> None:
        """
        Set the left/right child nodes for the current node

        Inputs:
            left  -> LHS child node
            right -> RHS child node
        """
        self.__left = left
        self.__right = right

    def get_left_node(self) -> Node:
        """
        Get the left child node

        Output:
            LHS child node
        """
        return (self.__left)

    def get_right_node(self) -> Node:
        """
        Get the RHS child node

        Output:
            RHS child node
        """
        return (self.__right)

class DecisionTree(ABC):
    """
    Base class to encompass the CART algorithm
    """

    def __init__(self, max_depth: int = None, min_samples_split: int = 2, nans_go_right=True, max_features=None) -> None:
        """
        Initializer

        Inputs:
            max_depth         -> maximum depth the tree can grow
            min_samples_split -> minimum number of samples required to split a node
            nans_go_right     -> boolean to determine where NaN values in predictors are allocated
        """
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.nans_go_right = nans_go_right
        self.max_features = max_features

    @abstractmethod
    def _impurity(self, D: np.array) -> None:
        """
        Protected function to define the impurity
        """
        pass

    @abstractmethod
    def _leaf_value(self, D: np.array) -> None:
        """
        Protected function to compute the value at a leaf node
        """
        pass

    def __split_right(self, feature_col: np.array, split_point: float) -> np.array:
        """
        Private function to determine which elements in the feature column go to the child right node

        Inputs:
            feature_col -> feature column to analyse
            split_point -> split point for the feature column
        Outputs:
            numpy array of boolean values
        """
        return (feature_col > split_point) | (np.isnan(feature_col) & self.nans_go_right)

    def __split_left(self, feature_col: np.array, split_point: float) -> np.array:
        """
        Private function to determine which elements in the feature column go to the child left node

        Inputs:
            feature_col -> feature column to analyse
            split_point -> split point for the feature column
        Outputs:
            numpy array of boolean values
        """
        return (feature_col <= split_point) | (np.isnan(feature_col) & (not self.nans_go_right))

    def __grow(self, node: Node, D: np.array, level: int) -> None:
        """
        Private recursive function to grow the tree during training

        Inputs:
            node  -> input tree node
            D     -> sample of data at node 
            level -> depth level in the tree for node
        """
        # are we in a leaf node?
        depth = (self.max_depth is None) or (self.max_depth >= (level+1))
        msamp = (self.min_samples_split <= D.shape[0])
        cls = D[:, -1]
        n_cls = np.unique(cls[~np.isnan(cls)]).shape[0] != 1

        # not a leaf node
        if depth and msamp and n_cls:

            #  max_features logic here
            if self.max_features is not None:
                if isinstance(self.max_features, float):
                    n_features = floor((D.shape[1] - 1) * self.max_features)
                else:
                    n_features = self.max_features
                available_features = list(range(D.shape[1]-1))
                selected_features = sample(available_features, min(n_features, len(available_features)))
            else:
                selected_features = range(D.shape[1]-1)

            # initialize the function parameters
            ip_node = None
            feature = None
            split = None
            left_D = None
            right_D = None
            # iterate through the possible feature/split combinations
            for f in selected_features:

                f_values = D[:, f]
                for s in np.unique(f_values[~np.isnan(f_values)]):
                    # for the current (f,s) combination, split the dataset
                    D_l = D[self.__split_left(D[:, f], s)]
                    D_r = D[self.__split_right(D[:, f], s)]
                    # ensure we have non-empty arrays
                    if D_l.size and D_r.size:
                        # calculate the impurity
                        ip = (D_l.shape[0]/D.shape[0])*self._impurity(D_l) + \
                            (D_r.shape[0]/D.shape[0])*self._impurity(D_r)
                        # now update the impurity and choice of (f,s)
                        if (ip_node is None) or (ip < ip_node):
                            ip_node = ip
                            feature = f
                            split = s
                            left_D = D_l
                            right_D = D_r

            # set the current node's parameters
            node.set_params(split, feature)

            # declare child nodes
            left_node = Node()
            right_node = Node()
            node.set_children(left_node, right_node)

            # investigate child nodes
            self.__grow(node.get_left_node(), left_D, level+1)
            self.__grow(node.get_right_node(), right_D, level+1)

        # is a leaf node
        else:

            # set the node value & return
            node.leaf_value = self._leaf_value(D)
            return
        
    def __traverse(self, node: Node, Xrow: np.array) -> int | float:
        
        """
        Private recursive function to traverse the (trained) tree

        Inputs:
            node -> current node in the tree
            Xrow -> data sample being considered
        Output:
            1rst version: leaf value corresponding to Xrow
        """
        # check if we're in a leaf node?
        if node.leaf_value is None:
            # get parameters at the node
            (s, f) = node.get_params()
            # decide to go left or right?
            if (self.__split_left(Xrow[f], s)):
                return (self.__traverse(node.get_left_node(), Xrow))
            else:
                # note nan's in Xrow will go right
                return (self.__traverse(node.get_right_node(), Xrow))
        else:
            # return the leaf value
            return (node.leaf_value)

    def train(self, Xin: np.array, Yin: np.array) -> None:
        """
        Train the CART model

        Inputs:
            #Xin -> input set of predictor features
            #Yin -> input set of labels
        """
        # prepare the input data
        D = np.concatenate((Xin, Yin.reshape(-1, 1)), axis=1)
        D[D == None] = np.nan
        D = D.astype('float64')

        # Calculate class weights (balanced)
        if self.class_weight == 'balanced':
            unique_classes, counts = np.unique(Yin, return_counts=True)
            n_samples = len(Yin)
            n_classes = len(unique_classes)
            self.class_weight = {cls: n_samples / (n_classes * count) for cls, count in zip(unique_classes, counts)}
            #self.class_weight = {cls: n_samples / count for cls, count in zip(unique_classes, counts)}

        # set the root node of the tree
        self.tree = Node()

        # build the tree
        self.__grow(self.tree, D, 1)

    def predict(self, Xin: np.array) -> np.array:
        """
        Make predictions from the trained CART model

        Input:
            Xin -> input set of predictor features
        Output:
            array of prediction values
        """
        # prepare the input data
        Xin[Xin == None] = np.nan
        Xin = Xin.astype('float64')
        # iterate through the rows of Xin
        p = []
        for r in range(Xin.shape[0]):
            p.append(self.__traverse(self.tree, Xin[r, :]))
        # return predictions
        return (np.array(p).flatten())


class DecisionTreeClassifier(DecisionTree):
    """
    Decision Tree Classifier
    """

    # def __init__(self, max_depth: int = None, min_samples_split: int = 2, nans_go_right=True, loss: str = 'gini') -> None:
        #"""
        #Initializer

        #Inputs:
           # max_depth         -> maximum depth the tree can grow
            #min_samples_split -> minimum number of samples required to split a node
            #nans_go_right     -> boolean to determine where NaN values in predictors are allocated
            #loss              -> loss function to use during training
        #"""
        #DecisionTree.__init__(
            #self, max_depth, min_samples_split, nans_go_right)
        #self.loss = loss

    def __init__(self, 
                 max_depth: int = None, 
                 min_samples_split: int = 2, 
                 nans_go_right=True, 
                 loss: str = 'gini', 
                 class_weight: dict = None, 
                 max_features=None) -> None: 
        
        super().__init__(max_depth, min_samples_split, nans_go_right, max_features)  
        self.loss = loss
        self.class_weight = class_weight if class_weight else {}


    def __gini(self, D: np.array) -> float:
        """
        Private function to define the gini impurity

        Input:
            D -> data to compute the gini impurity over
        Output:
            Gini impurity for D
        """
        G = 0
        # iterate through the unique classes
        cls = D[:, -1]
        for c in np.unique(cls[~np.isnan(cls)]):

            # compute p for the current c
            p = D[D[:, -1] == c].shape[0] / D.shape[0]
            w = self.class_weight.get(c, 1) 

            # compute term for the current c multiplied by the class weight
            G += w * p * (1 - p)  
        # return gini impurity
        return G



    def __entropy(self, D: np.array) -> float:

        """
        Private function to define the shannon entropy

        Input:
            D -> data to compute the shannon entropy over
        Output:
            Shannon entropy for D
        """
        # initialize the output
        H = 0
        # iterate through the unique classes
        cls = D[:, -1]

        for c in np.unique(cls[~np.isnan(cls)]):

            # compute p for the current c
            p = D[D[:, -1] == c].shape[0] / D.shape[0]

            # compute term for the current c multiplied by the class weight
            w = self.class_weight.get(c, 1)  
            H -= w * p * np.log2(p)  
        # return entropy
        return H

    def _impurity(self, D: np.array) -> float:
        """
        Protected function to define the impurity

        Input:
            D -> data to compute the impurity metric over
        Output:
            Impurity metric for D        
        """
        # use the selected loss function to calculate the node impurity
        ip = None
        if self.loss == 'gini':
            ip = self.__gini(D)
        elif self.loss == 'entropy':
            ip = self.__entropy(D)
        # return results
        return (ip)

    def _leaf_value(self, D: np.array) -> int:
        """
        Protected function to compute the value at a leaf node

        Input:
            D -> data to compute the leaf value
        Output:
            Mode of D         
        """ 
        return (stats.mode(D[:, -1], keepdims=False, nan_policy='omit')[0])