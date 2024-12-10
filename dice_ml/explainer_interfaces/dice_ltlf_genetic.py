"""
Module to generate diverse counterfactual explanations based on genetic algorithm
This code is similar to 'GeCo: Quality Counterfactual Explanations in Real Time': https://arxiv.org/pdf/2101.01292.pdf
"""
import copy
import random
import timeit
import numpy as np
import pandas as pd
import os
from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml.utils.exception import UserConfigValidationException
from pm4py import convert_to_event_log,format_dataframe
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessModels.LTLModel import LTLModel,LTLTemplate
from Declare4Py.ProcessMiningTasks.ConformanceChecking.LTLAnalyzer import LTLAnalyzer
import re
from scipy.spatial.distance import _validate_vector
from scipy.spatial.distance import cdist, pdist
from scipy.stats import median_abs_deviation
from logaut import ltl2dfa
from Declare4Py.ProcessMiningTasks.ConformanceChecking.LTLAnalyzer import run_single_trace, is_sink
from Declare4Py.Utils.utils import *
from functools import reduce
import itertools
from datetime import datetime
from nirdizati_light.explanation.visualizations.utils import *
import logging
import time

class DiceGeneticLTLf(ExplainerBase):

    def __init__(self, data_interface, model_interface, encoder=None, dataset=None):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        super().__init__(data_interface, model_interface)  # initiating data related parameters
        self.num_output_nodes = None

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''

        # Initializing a label encoder to obtain label-encoded values for categorical variables
        self.labelencoder = set()
        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

    def update_hyperparameters(self, proximity_weight, sparsity_weight, plausibility_weight,
                               diversity_weight, categorical_penalty, conformance_weight):
        """Update hyperparameters of the loss function"""

        self.proximity_weight = proximity_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty
        self.plausibility_weight = plausibility_weight
        self.conformance_weight = conformance_weight

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights,
                                encoding='one-hot'):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]
        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type
        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=False)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1 / normalized_mads[feature], 2)

            feature_weights_list = []
            if encoding == 'one-hot':
                for feature in self.data_interface.encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
            elif encoding == 'label':
                for feature in self.data_interface.feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        # the weight is inversely proportional to max value
                        feature_weights_list.append(round(1 / self.feature_range[feature].max(), 2))
            self.feature_weights_list = [feature_weights_list]

    # make do_random_init function more efficient

    def do_random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range):
        valid_inits = []
        precisions = self.data_interface.get_decimal_precisions()

        while len(valid_inits) < num_inits:
            num_remaining = num_inits - len(valid_inits)
            num_features = self.data_interface.number_of_features

            # Generate random initializations for all features at once
            random_inits = np.zeros((num_remaining, num_features))
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature in features_to_vary:
                    if feature in self.data_interface.continuous_feature_names:
                        random_inits[:, jx] = self.rng.uniform(self.feature_range[feature][0],
                                                                self.feature_range[feature][1], num_remaining)
                        random_inits[:, jx] = np.round(random_inits[:, jx], precisions[jx])
                    else:
                        random_inits[:, jx] = self.rng.choice(self.feature_range[feature], num_remaining)
                else:
                    random_inits[:, jx] = query_instance[jx]

            # Filter out the valid initializations
            valid_mask = np.apply_along_axis(self.is_cf_valid, 1, self.predict_fn_scores(random_inits))
            valid_inits.extend(random_inits[valid_mask])

        return np.array(valid_inits[:num_inits])

    def do_KD_init(self, features_to_vary, query_instance, cfs, desired_class, desired_range):
        # cfs = self.label_encode(cfs)
        # cfs = pd.DataFrame(cfs,columns=self.data_interface.feature_names)
        cfs = cfs.reset_index(drop=True)
        query_instance = query_instance.reshape(-1, 1)
        self.cfs = np.zeros((self.population_size, self.data_interface.number_of_features))
        for kx in range(self.population_size):
            if kx >= len(cfs):
                break
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature not in features_to_vary:
                    one_init[jx] = (query_instance[jx])
                else:
                    if feature in self.data_interface.continuous_feature_names:
                        if self.feature_range[feature][0] <= cfs.iat[kx, jx] <= self.feature_range[feature][1]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if self.feature_range[feature][0] <= query_instance[jx] <= self.feature_range[feature][1]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = self.rng.uniform(
                                    self.feature_range[feature][0], self.feature_range[feature][1])
                    else:
                        if float(cfs.iat[kx, jx]) in self.feature_range[feature]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if float(query_instance[jx]) in self.feature_range[feature]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = self.rng.choice(self.feature_range[feature])
            self.cfs[kx] = one_init
            kx += 1

        new_array = [tuple(row) for row in self.cfs]
        uniques = np.unique(new_array, axis=0)

        if len(uniques) != self.population_size:
            remaining_cfs = self.do_random_init(
                self.population_size - len(uniques), features_to_vary, query_instance, desired_class, desired_range)
            self.cfs = np.concatenate([uniques, remaining_cfs])

    def do_cf_initializations(self, total_CFs, initialization, algorithm, features_to_vary, desired_range,
                              desired_class,
                              query_instance, query_instance_df_dummies, verbose):
        """Intializes CFs and other related variables."""
        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1  # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        self.features_to_vary = features_to_vary

        # CF initialization
        self.cfs = []
        if initialization == 'random':
            self.cfs = self.do_random_init(
                self.population_size, features_to_vary, query_instance, desired_class, desired_range)

        elif initialization == 'kdtree':
            # Partitioned dataset and KD Tree for each class (binary) of the dataset
            self.dataset_with_predictions, self.KD_tree, self.predictions = \
                self.build_KD_tree(self.data_interface.data_df.copy(),
                                   desired_range, desired_class, self.predicted_outcome_name)
            if self.KD_tree is None:
                self.cfs = self.do_random_init(
                    self.population_size, features_to_vary, query_instance, desired_class, desired_range)

            else:
                num_queries = min(len(self.dataset_with_predictions), self.population_size * self.total_CFs)
                indices = self.KD_tree.query(query_instance_df_dummies, num_queries)[1][0]
                KD_tree_output = self.dataset_with_predictions.iloc[indices].copy()
                self.do_KD_init(features_to_vary, query_instance, KD_tree_output, desired_class, desired_range)

        if verbose:
            print("Initialization complete! Generating counterfactuals...")

    def do_param_initializations(self, total_CFs, initialization, desired_range, desired_class,
                                 query_instance, query_instance_df_dummies, algorithm, features_to_vary,
                                 permitted_range, yloss_type, diversity_loss_type, feature_weights,
                                 proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
                                 categorical_penalty, conformance_weight, verbose):
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.feature_range = self.get_valid_feature_range(normalized=False)
        if len(self.cfs) != total_CFs:
            self.do_cf_initializations(
                total_CFs, initialization, algorithm, features_to_vary, desired_range, desired_class,
                query_instance, query_instance_df_dummies, verbose)
        else:
            self.total_CFs = total_CFs
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        self.update_hyperparameters(proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
                                    categorical_penalty, conformance_weight)

    def _generate_counterfactuals(self, query_instance, total_CFs, encoder, dataset, initialization="kdtree",
                                  desired_range=None, desired_class="opposite", proximity_weight=0.5,
                                  sparsity_weight=0.5, diversity_weight=0.5, plausibility_weight=0.5,
                                  categorical_penalty=0.1,
                                  algorithm="DiverseCF", features_to_vary="all", permitted_range=None,
                                  yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", stopping_threshold=0.25, posthoc_sparsity_param=0,
                                  posthoc_sparsity_algorithm="linear", maxiterations=50, thresh=1e-2, verbose=True,
                                  conformance_weight=3,
                                  model_path=None, optimization=None, heuristic=None, random_seed=None, adapted=None,ltlf_model=None,
                                  original_activities=None, dfa = None):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param initialization: Method to use to initialize the population of the genetic algorithm
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: For classification problems. Desired counterfactual class - can take 0 or 1.
                              Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                                Defaults to the range inferred from training data. If None, uses the parameters initialized
                                in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and
                                corresponding weights as values. Default option is "inverse_mad" where the
                                weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD)
                                of the feature's values in the training set; the weight for a categorical feature is
                                equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param maxiterations: Maximum iterations to run the genetic algorithm for.
        :param thresh: The genetic algorithm stops when the difference between the previous best loss and current
                       best loss is less than thresh
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                 (see diverse_counterfactuals.py).
        """
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        if not hasattr(self.data_interface, 'data_df') and initialization == "kdtree":
            raise UserConfigValidationException(
                "kd-tree initialization is not supported for private data"
                " interface because training data to build kd-tree is not available.")

        self.population_size = 5 * total_CFs

        self.start_time = timeit.default_timer()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        # Prepares user defined query_instance for DiCE.
        query_instance_orig = query_instance
        query_instance_orig = self.data_interface.prepare_query_instance(
            query_instance=query_instance_orig)
        query_instance = self.data_interface.prepare_query_instance(
            query_instance=query_instance)
        # number of output nodes of ML model
        self.num_output_nodes = None
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = self.model.get_num_output_nodes2(query_instance)

        # query_instance = self.label_encode(query_instance)
        # query_instance = pd.DataFrame(query_instance,columns=self.data_interface.feature_names)
        query_instance = np.array(query_instance.values[0])
        self.x1 = query_instance
        query_instance = query_instance.reshape(1, -1)
        # find the predicted value of query_instance
        test_pred = self.predict_fn(query_instance)

        self.test_pred = test_pred

        desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred)

        query_instance_df_dummies = pd.get_dummies(query_instance_orig)
        for col in self.data_interface.get_all_dummy_colnames():
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance,
                                      query_instance_df_dummies, algorithm, features_to_vary, permitted_range,
                                      yloss_type, diversity_loss_type, feature_weights, proximity_weight,
                                      sparsity_weight, plausibility_weight, diversity_weight, categorical_penalty,
                                      conformance_weight, verbose)
        #d4py = Declare4Py()
        #d4py.parse_decl_model(os.path.join(model_path, (dataset + '.decl')))
        #self.filter_declare_model(query_instance, encoder, d4py)

#        activities, activations, targets = self.get_constraint_activities(ltlf_model)
        #activities = ltlf_model.parameters
        activities = list(original_activities.values())
        activations = activities
        targets=activities
        #activations =  [i.argument.name.split('_')[1] for i in ltlf_model.parsed_formula.operands]
        #targets = [i.argument.name.split('_')[1] for i in ltlf_model.parsed_formula.operands]
        query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary,
                                                      maxiterations, thresh, verbose, encoder, dataset, model_path,
                                                      ltlf_model, optimization,
                                                      heuristic, activities, activations, targets, adapted, original_activities, dfa)
        iterations = self.number_of_iterations
        with open("../experiments/ltlfcf_new_results_apriori_dfa/iterations_count.txt", "a") as file:
            file.write(f"Number of iterations: {dataset,heuristic,total_CFs,iterations}\n")

        ## change model given to this function
        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          test_instance_df=query_instance_df,
                                          final_cfs_df=self.final_cfs_df,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class,
                                          model_type=self.model.model_type)

    def predict_fn_scores(self, input_instance):
        """Returns prediction scores."""
        # input_instance = self.label_decode(input_instance)
        input_instance = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        out = self.model.get_output(input_instance)
        if self.model.model_type == ModelTypes.Classifier and out.shape[1] == 1:
            # DL models return only 1 for binary classification
            out = np.hstack((1 - out, out))
        return out

    def predict_fn(self, input_instance):
        """Returns actual prediction."""
        # input_instance = self.label_decode(input_instance)
        input_instance = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        preds = self.model.get_output(input_instance, model_score=False)
        return preds

    def _predict_fn_custom(self, input_instance, desired_class):
        """Checks that the maximum predicted score lies in the desired class."""
        """The reason we do so can be illustrated by
        this example: If the predict probabilities are [0, 0.5, 0,5], the computed yloss is 0 as class 2 has the same
        value as the maximum score. sklearn's usual predict function, which implements argmax, returns class 1 instead
        of 2. This is why we need a custom predict function that returns the desired class if the maximum predict
        probability is the same as the probability of the desired class."""

        # input_instance = self.label_decode(input_instance)
        input_instance = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        output = self.model.get_output(input_instance, model_score=True)
        if self.model.model_type == ModelTypes.Classifier and np.array(output).shape[1] == 1:
            # DL models return only 1 for binary classification
            output = np.hstack((1 - output, output))
        desired_class = int(desired_class)
        maxvalues = np.max(output, 1)
        predicted_values = np.argmax(output, 1)

        # We iterate through output as we often call _predict_fn_custom for multiple inputs at once
        for i in range(len(output)):
            if output[i][desired_class] == maxvalues[i]:
                predicted_values[i] = desired_class

        return predicted_values

    def compute_plausibility(self, cfs=None, ratio_cont=None):
        query_instance = self.x1
        continuous_features = self.data_interface.continuous_feature_names
        categorical_features = self.data_interface.categorical_feature_names
        dists = []
        ratio_cont = len(continuous_features) / len(categorical_features)
        X_y = self.data_interface.data_df
        if cfs is None:
            cfs = self.cfs
        for cf in cfs:
            neigh_dist = self.distance_mh(query_instance=query_instance.reshape(1, -1), cf_list=cfs, X=X_y)
            idx_neigh = np.argsort(neigh_dist)[0]
            closest = X_y.to_numpy()[idx_neigh]
            d = self.distance_mh(query_instance=cf.reshape(1, -1), cf_list=closest.reshape(1, -1), X=X_y)
            dists.append(d)
        return np.array(dists)

    # update here to not get confused
    def distance_mh(self, query_instance, cf_list, X, ratio_cont=None, agg=None):
        nbr_features = self.data_interface.number_of_features
        cont_feature_index = self.data_interface.continuous_feature_indexes
        cat_feature_index = self.data_interface.categorical_feature_indexes
        dist_cont = self.continuous_distance(query_instance=query_instance, cf_list=cf_list, metric='mad', X=X, agg=agg)
        try:
            dist_cate = self.categorical_distance(query_instance=query_instance.astype('float64'), cf_list=cf_list, metric='hamming', agg=agg)
        except:
            dist_cate = self.categorical_distance(query_instance=query_instance.astype('float64'), cf_list=cf_list.astype('float64'), metric='hamming', agg=agg)

        if ratio_cont is None:
            ratio_continuous = len(cont_feature_index) / nbr_features
            ratio_categorical = len(cat_feature_index) / nbr_features
        else:
            ratio_continuous = ratio_cont
            ratio_categorical = 1.0 - ratio_cont
        dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
        return dist

    def continuous_distance(self, query_instance, cf_list, metric='euclidean', X=None, agg=None):
        cont_feature_index = self.data_interface.continuous_feature_indexes
        if metric == 'mad':
            mad = median_abs_deviation(X.iloc[:, cont_feature_index], axis=0)
            mad = np.array([v if v != 0 else 1.0 for v in mad])

            def _mad_cityblock(u, v):
                return mad_cityblock(u, v, mad)

            dist = cdist(query_instance.reshape(1, -1)[:, cont_feature_index].astype('float'),
                         cf_list[:, cont_feature_index].astype('float'), metric=_mad_cityblock)
        else:
            dist = cdist(query_instance.reshape(1, -1)[:, cont_feature_index].astype('float'),
                         cf_list[:, cont_feature_index].astype('float'), metric=metric)

        if agg is None or agg == 'mean':
            return np.mean(dist)

        if agg == 'max':
            return np.max(dist)

        if agg == 'min':
            return np.min(dist)

    def categorical_distance(self, query_instance, cf_list, metric='jaccard', agg=None):
        cat_feature_index = self.data_interface.categorical_feature_indexes
        dist = cdist(query_instance.reshape(1, -1)[:, cat_feature_index], cf_list[:, cat_feature_index], metric=metric)

        if agg is None or agg == 'mean':
            return np.mean(dist)

        if agg == 'max':
            return np.max(dist)

        if agg == 'min':
            return np.min(dist)

    def compute_yloss(self, cfs, desired_range, desired_class):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.model.model_type == ModelTypes.Classifier:
            predicted_value = np.array(self.predict_fn_scores(cfs))
            if self.yloss_type == 'hinge_loss':
                maxvalue = np.full((len(predicted_value)), -np.inf)
                for c in range(self.num_output_nodes):
                    if c != desired_class:
                        maxvalue = np.maximum(maxvalue, predicted_value[:, c])
                yloss = np.maximum(0, maxvalue - predicted_value[:, int(desired_class)])
            return yloss

        elif self.model.model_type == ModelTypes.Regressor:
            predicted_value = self.predict_fn(cfs)
            if self.yloss_type == 'hinge_loss':
                yloss = np.zeros(len(predicted_value))
                for i in range(len(predicted_value)):
                    if not desired_range[0] <= predicted_value[i] <= desired_range[1]:
                        yloss[i] = min(abs(predicted_value[i] - desired_range[0]),
                                       abs(predicted_value[i] - desired_range[1]))
            return yloss

    def compute_proximity_loss(self, x_hat_unnormalized, query_instance_normalized):
        """Compute weighted distance between two vectors."""
        x_hat = self.data_interface.normalize_data(x_hat_unnormalized)
        feature_weights = np.array(
            [self.feature_weights_list[0][i] for i in self.data_interface.continuous_feature_indexes])
        product = np.multiply(
            (abs(x_hat - query_instance_normalized)[:, [self.data_interface.continuous_feature_indexes]]),
            feature_weights)
        product = product.reshape(-1, product.shape[-1])
        proximity_loss = np.sum(product, axis=1)

        # Dividing by the sum of feature weights to normalize proximity loss
        return proximity_loss / sum(feature_weights)

    def compute_sparsity_loss(self, cfs):
        """Compute weighted distance between two vectors."""
        sparsity_loss = np.count_nonzero(np.asarray(cfs, dtype='int') - np.asarray(self.x1, dtype='int'), axis=1)
        return sparsity_loss / len(
            self.data_interface.feature_names)  # Dividing by the number of features to normalize sparsity loss

    def compute_filtered_loss(self, query_instance, cfs, desired_range, desired_class):
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 and len(self.data_interface.continuous_feature_indexes) > 1 else 0.0
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        self.plausibility_loss = self.compute_plausibility(cfs=cfs)
        # TODO DO ONLY ONE ROUND OF COMPUTE PERFORMANCE AND SAVE THE SCORE
        self.loss = np.reshape(np.array(self.yloss
                                        + (self.proximity_weight * self.proximity_loss) + (
                                                    self.sparsity_weight * self.sparsity_loss)
                                        # + (self.conformance_weight * (1 - self.conformance_score))
                                        + (self.plausibility_weight * self.plausibility_loss)), (-1, 1))
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss

    def compute_baseline_loss(self, query_instance, cfs, desired_range, desired_class):
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 and len(self.data_interface.continuous_feature_indexes) > 1 else 0.0
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        self.plausibility_loss = self.compute_plausibility(cfs=cfs)
        # TODO DO ONLY ONE ROUND OF COMPUTE PERFORMANCE AND SAVE THE SCORE
        self.loss = np.reshape(np.array(self.yloss
                                        + (self.proximity_weight * self.proximity_loss) + (
                                                    self.sparsity_weight * self.sparsity_loss)
                                        + (self.plausibility_weight * self.plausibility_loss)), (-1, 1))
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss

    def compute_loss(self, query_instance, cfs, desired_range, desired_class):
        """Computes the overall loss"""
        ##TODO Fix proximity loss
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 and len(self.data_interface.continuous_feature_indexes) > 1 else 0.0
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else 0.0
        self.plausibility_loss = self.compute_plausibility(cfs=cfs)
        # TODO DO ONLY ONE ROUND OF COMPUTE PERFORMANCE AND SAVE THE SCORE
        self.loss = np.reshape(np.array(self.yloss
                                        + (self.proximity_weight * self.proximity_loss) + (
                                                self.sparsity_weight * self.sparsity_loss) +
                                        + (self.conformance_weight * (1 - self.conformance_score))
                                        + (self.plausibility_weight * self.plausibility_loss)), (-1, 1))

        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss


    def mate(self, k1, k2, features_to_vary, query_instance):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        #rng.bit_generator.state = np.random.PCG64(self.random_seed).state

        one_init = np.zeros(self.data_interface.number_of_features)
        for j in range(self.data_interface.number_of_features):
            gp1 = k1[j]
            gp2 = k2[j]
            feat_name = self.data_interface.feature_names[j]
            # random probability
            prob = self.rng.random()
            if prob < 0.5:
                # if prob is less than 0.40, insert gene from parent 1
                one_init[j] = gp1
            elif prob < 0.85:
                # if prob is between 0.40 and 0.80, insert gene from parent 2
                one_init[j] = gp2
            else:
                # otherwise insert random gene(mutate) for maintaining diversity
                if feat_name in features_to_vary:
                    if feat_name in self.data_interface.continuous_feature_names:
                        one_init[j] = self.rng.uniform(self.feature_range[feat_name][0],
                                                        self.feature_range[feat_name][0])
                    else:
                        one_init[j] = self.rng.choice(self.feature_range[feat_name])
                else:
                    one_init[j] = query_instance[j]
        return one_init
    ### SHIFT FROM PARENT CONFORMANCE TO QUERY_INSTANCE CONFORMANCE
    # d4py.get_model_activities()
    # mate_2 represents the second heuristic, where we relax the contraints to include the targets that may occur
    # in the future
    # mate_1 represents the first heuristic where we do not use the activities again
    def mate_1(self, k1, k2, features_to_vary, query_instance, encoder, activities):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        original_query_df = pd.DataFrame(np.asarray(query_instance, dtype=float),
                                         columns=self.data_interface.feature_names)
        k1df = pd.DataFrame([k1], columns=self.data_interface.feature_names)
        k2df = pd.DataFrame([k2], columns=self.data_interface.feature_names)
        encoder.decode(k1df)
        encoder.decode(k2df)
        encoder.decode(original_query_df)
        one_init = np.zeros(self.data_interface.number_of_features)

        filter_query = original_query_df[original_query_df.isin(activities)]
        child = filter_query[filter_query.notnull()]
        child = child.to_numpy().reshape(-1)

        for j in range(self.data_interface.number_of_features):
            feat_name = self.data_interface.feature_names[j]
            prob = self.rng.random()
            if ('prefix' in feat_name) & (pd.isnull(child[j])):
                if (k1df[feat_name][0] not in activities) and (prob < 0.5):
                    child[j] = k1df[feat_name][0]
                elif (k2df[feat_name][0] not in activities) and (prob < 0.85):
                    child[j] = k2df[feat_name][0]
                else:
                    child[j] = self.rng.choice(
                        [x for x in encoder._label_dict[feat_name].keys() if x not in activities])
            elif 'prefix' not in feat_name:
                gp1 = k1[j]
                gp2 = k2[j]
                if prob < 0.5:
                    # if prob is less than 0.40, insert gene from parent 1
                    child[j] = gp1
                elif prob < 0.85:
                    # if prob is between 0.40 and 0.80, insert gene from parent 2
                    child[j] = gp2
                else:
                    # otherwise insert random gene(mutate) for maintaining diversity
                    if feat_name in features_to_vary:
                        if feat_name in self.data_interface.continuous_feature_names:
                            child[j] = self.rng.uniform(self.feature_range[feat_name][0],
                                                         self.feature_range[feat_name][1])
                        else:
                            child[j] = self.rng.choice(self.feature_range[feat_name])
                    else:
                        child[j] = query_instance[j]
            else:
                pass

        child = pd.DataFrame([child], columns=self.data_interface.feature_names)
        encoder.encode(child)
        return child


    def mate_2(self, k1, k2, features_to_vary, query_instance, encoder, activities, dfa,backend,
               original_activities):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        k1df = pd.DataFrame([k1], columns=self.data_interface.feature_names)
        k2df = pd.DataFrame([k2], columns=self.data_interface.feature_names)
        original_query_df = pd.DataFrame(np.asarray(query_instance, dtype=float),
                                         columns=self.data_interface.feature_names)
        encoder.decode(k1df)
        encoder.decode(k2df)
        encoder.decode(original_query_df)
        one_init = np.zeros(self.data_interface.number_of_features)
        filter_query = original_query_df[original_query_df.isin(activities)]
        child = filter_query[filter_query.notnull()]
        child = child.to_numpy().reshape(-1)
        ######
        ####
        for j in range(self.data_interface.number_of_features):
            feat_name = self.data_interface.feature_names[j]
            prob = self.rng.random()
            if ('prefix' in feat_name) and (pd.isnull(child[j])):
                if (k1df[feat_name][0] not in activities) and (prob < 0.50):
                    child[j] = k1df[feat_name][0]
                elif (k2df[feat_name][0] not in activities) and (prob < 0.85):
                    child[j] = k2df[feat_name][0]
                #else:
                #    child[j] = original_query_df.iloc[0,j]
            elif ('prefix' in feat_name) and prob >= 0.85:
                    # Take child up to j
                    # Check state in DFA, make function to the check state of input, return activities to exclude
                    # From transition function extract activities to exclude from mutations
                include,exclude = check_state(child[:j],dfa,singleton,backend,original_activities,
                                              child[j])
#                   print(j,'set',[x for x in encoder._label_dict[feat_name].keys()
#                                     if x not in exclude and (x in include or x not in include)])
#                   print('choice',[x for x in encoder._label_dict[feat_name].keys()
#                                     if x not in exclude and (x in include or x not in include)])
                if '*' in include:
                    include.remove('*')
                    child[j] = self.rng.choice(list(set([x for x in encoder._label_dict[feat_name].keys()
                            if x not in exclude]+ include)))
                else:
                    child[j] = self.rng.choice(include)
                #child[j] = self.rng.choice(list(set([x for x in encoder._label_dict[feat_name].keys()
                #            if x not in exclude]+ include)))
            elif 'prefix' not in feat_name:
                gp1 = k1[j]
                gp2 = k2[j]
                if prob < 0.40:
                    # if prob is less than 0.40, insert gene from parent 1
                    child[j] = gp1
                elif prob < 0.80:
                    # if prob is between 0.40 and 0.80, insert gene from parent 2
                    child[j] = gp2
                else:
                    # otherwise insert random gene(mutate) for maintaining diversity
                    if feat_name in features_to_vary:
                        if feat_name in self.data_interface.continuous_feature_names:
                            child[j] = self.rng.uniform(self.feature_range[feat_name][0],
                                                         self.feature_range[feat_name][1])
                        else:
                            child[j] = self.rng.choice(self.feature_range[feat_name])
                    else:
                        child[j] = query_instance[j]
            else:
                child[j] = original_query_df.iloc[0,j]
        child = pd.DataFrame([child], columns=self.data_interface.feature_names)
        encoder.encode(child)
        return child
    def mate_5(self, k1, k2, features_to_vary, query_instance, encoder, dfa, activities, activations,
               targets,
               original_activities, backend, max_retries=100, skip=True):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        original_query_df = pd.DataFrame(np.asarray(query_instance, dtype=float),
                                         columns=self.data_interface.feature_names)
        k1df = pd.DataFrame([k1], columns=self.data_interface.feature_names)
        k2df = pd.DataFrame([k2], columns=self.data_interface.feature_names)
        #encoder.decode(k1df)
        #encoder.decode(k2df)
        # If you want to preserve the columns structure from filter_query, you can initialize it like this:
        one_init = np.zeros(self.data_interface.number_of_features)
        child = pd.DataFrame([one_init],columns=original_query_df.columns)
        child = child.to_numpy().reshape(-1)
        random_choice_indices = []
        mutation = 0
        mistakes = 0
        # Keep track of where the random choice operation was performed
        for j in range(self.data_interface.number_of_features):
            feat_name = self.data_interface.feature_names[j]
            prob = self.rng.random()
            gp1 = k1df[feat_name][0]
            gp2 = k2df[feat_name][0]
            if prob < 0.40:
                # if prob is less than 0.40, insert gene from parent 1
                child[j] = gp1
            elif prob < 0.80:
                # if prob is between 0.40 and 0.80, insert gene from parent 2
                child[j] = gp2
            else:
                mutation += 1
                # otherwise insert random gene(mutate) for maintaining diversity
                if feat_name in features_to_vary:
                    child[j] = self.rng.choice(self.feature_range[feat_name])
                    random_choice_indices.append(j)
                else:
                    child[j] = query_instance[j]
        retries = 0
        while retries < max_retries:
            child_check = pd.DataFrame([child], columns=self.data_interface.feature_names)
            encoder.decode(child_check)
            child_check = child_check.to_numpy().reshape(-1)
            if backend == 'ltlf2dfa':
                current_states = {1}
            else:
                current_states = {dfa.initial_state}
            for event in child_check:
                temp = dict()
                symbol = str(event)
                symbol = Utils.parse_parenthesis(symbol)
                if not skip:
                    symbol = Utils.encode_attribute_type(attribute) + "_" + symbol
                symbol = Utils.parse_activity(symbol)
                symbol = symbol.lower()
                temp[symbol] = True
                result_set = set()
                for x in current_states:
                    successors = dfa.get_successors(x, temp)
                    current_states = result_set.union(successors)

            is_accepted = any(dfa.is_accepting(state) for state in current_states)
            if is_accepted:
                child = pd.DataFrame([child], columns=self.data_interface.feature_names)
                encoder.encode(child)
                return child
            else:
                if len(random_choice_indices) > 0:
                    mistakes += 1
                    child[random_choice_indices] = [self.rng.choice(
                        [x for x in encoder._label_dict[self.data_interface.feature_names[i]].values()]) for i in
                        random_choice_indices]

            retries += 1
        # Return the child as it is if no valid solution is found within the retry limit
        child = pd.DataFrame([child], columns=self.data_interface.feature_names)
        encoder.encode(child)
        return child
    def find_counterfactuals(self, query_instance, desired_range, desired_class,
                             features_to_vary, maxiterations, thresh, verbose, encoder, dataset, model_path, ltlf_model,
                             optimization,
                             heuristic, activities, activations, targets, adapted,original_activities, dfa):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""
        population = self.cfs.copy()
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        cfs_preds = [np.inf] * self.total_CFs
        to_pred = None

        self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
        self.query_instance_normalized = self.query_instance_normalized.astype('float')
        while iterations < maxiterations and self.total_CFs > 0:
            if abs(previous_best_loss - current_best_loss) <= thresh and \
                    (self.model.model_type == ModelTypes.Classifier and all(i == desired_class for i in cfs_preds) or
                     (self.model.model_type == ModelTypes.Regressor and
                      all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
                stop_cnt += 1
            else:
                stop_cnt = 0
            if stop_cnt >= 5:
                break
            previous_best_loss = current_best_loss
            population = np.unique(tuple(map(tuple, population)), axis=0)
            self.conformance_score, population_conformance = self.compute_conformance_new(population, encoder, ltlf_model, dfa)
            population_fitness = self.compute_loss(query_instance, population, desired_range, desired_class)
            # elif (optimization == 'loss_function') & (iterations > 0) & (not adapted):

            population_fitness = population_fitness[population_fitness[:, 1].argsort()]
            current_best_loss = population_fitness[0][1]
            to_pred = np.array([population[int(tup[0])] for tup in population_fitness[:self.total_CFs]])

            if self.total_CFs > 0:
                if self.model.model_type == ModelTypes.Classifier:
                    cfs_preds = self._predict_fn_custom(to_pred, desired_class)
                else:
                    cfs_preds = self.predict_fn(to_pred)



            # self.total_CFS of the next generation obtained from the fittest members of current generation
            top_members = self.total_CFs
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])

            #DFA AND SINGLETON USED FOR HEURISTIC 2,3,4
            backend = 'ltlf2dfa'
           # try:
           #     dfa = ltl2dfa(ltlf_model.parsed_formula,backend=backend)
           # except:
           #     dfa = ltl2dfa(ltlf_model.parsed_formula,backend=backend)
           #     print('Problem')
           #     print("model",ltlf_model.parsed_formula)
            letters = alphabet(parse(ltlf_model.formula))
            letters.add('*')
            # rest of the next generation obtained from top 50% of fittest members of current generation
            rest_members = self.population_size - top_members

            new_generation_2 = None
            if rest_members > 0:
                new_generation_2 = np.zeros((rest_members, self.data_interface.number_of_features))
                for new_gen_idx in range(rest_members):
                    idx1 = self.rng.integers(0, int(len(population) / 2))
                    parent1 = population[idx1]
                    idx2 = self.rng.integers(0, int(len(population) / 2))
                    while idx1 == idx2:
                        idx2 = self.rng.integers(0, int(len(population) / 2))
                    parent2 = population[idx2]
                    # if heuristic == 'heuristic_1':
                    #     child = self.mate_1(parent1, parent2, features_to_vary, query_instance,encoder,d4py,activities,activations,targets)
                    if adapted:
                        if heuristic == 'heuristic_1':
                            child = self.mate_1(parent1, parent2, features_to_vary, query_instance, encoder,
                                                activities)
                        elif heuristic == 'heuristic_2':
                            child = self.mate_2(parent1, parent2, features_to_vary, query_instance, encoder,activities,
                                                dfa, backend, original_activities)
                        elif heuristic == 'mar':
                            #This heuristic entails performing the mutation and crossover, reaching the end of the prefix
                            #and then checking the state of the automaton to see if we are in a final state
                            #if yes, we move on, otherwise we reperform the mutation operation on the gene we mutated
                            #until we reach a final state
                            self.mistakes = []
                            child = self.mate_5(parent1, parent2, features_to_vary, query_instance, encoder, dfa,
                                                activities, activations, targets, original_activities, backend)
                    else:
                        child = self.mate(parent1, parent2, features_to_vary, query_instance)

                    new_generation_2[new_gen_idx] = child

            if new_generation_2 is not None:
                # if self.total_CFs > 0:
                population = np.concatenate([new_generation_1, new_generation_2])
                # else:
                #    population = new_generation_2
            else:
                raise SystemError("The number of total_Cfs is greater than the population size!")
            iterations += 1

        self.cfs_preds = []
        self.final_cfs = []
        i = 0
        while i < self.population_size and i < len(population):
            predictions = self.predict_fn_scores(population[i].reshape(1, -1))[0]
            if self.is_cf_valid(predictions):
                self.final_cfs.append(population[i])
                # checking if predictions is a float before taking the length as len() works only for array-like
                # elements. isinstance(predictions, (np.floating, float)) checks if it's any float (numpy or otherwise)
                # We do this as we take the argmax if the prediction is a vector -- like the output of a classifier
                if not isinstance(predictions, (np.floating, float)) and len(predictions) > 1:
                    self.cfs_preds.append(np.argmax(predictions))
                else:
                    self.cfs_preds.append(predictions)
            if len(self.final_cfs) >= self.total_CFs:
                break
            i += 1

        # converting to dataframe
        # query_instance_df = self.label_decode(query_instance)
        query_instance_df = pd.DataFrame(query_instance, columns=self.data_interface.feature_names)
        query_instance_df[self.data_interface.outcome_name] = self.test_pred
        self.final_cfs_df = pd.DataFrame(self.final_cfs, columns=self.data_interface.feature_names)
        self.final_cfs_df_sparse = copy.deepcopy(self.final_cfs_df)

        if self.final_cfs_df is not None:
            self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
            self.final_cfs_df_sparse[self.data_interface.outcome_name] = self.cfs_preds
            self.round_to_precision()

        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            if len(self.final_cfs) == self.total_CFs:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) ' % (len(self.final_cfs), self.total_CFs),
                      'Diverse Counterfactuals found for the given configuation, perhaps ',
                      'change the query instance or the features to vary...'  '; total time taken: %02d' % m,
                      'min %02d' % s, 'sec')
        self.number_of_iterations = iterations

        return query_instance_df

    def label_encode(self, input_instance):
        for column in self.data_interface.categorical_feature_names:
            input_instance[column] = self.labelencoder[column].transform(input_instance[column])
        return input_instance

    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values
        """
        num_to_decode = 1
        if len(labelled_input.shape) > 1:
            num_to_decode = len(labelled_input)
        else:
            labelled_input = [labelled_input]

        input_instance = []

        for j in range(num_to_decode):
            temp = {}
            for i in range(len(labelled_input[j])):
                if self.data_interface.feature_names[i] in self.data_interface.categorical_feature_names:
                    enc = self.labelencoder[self.data_interface.feature_names[i]]
                    val = enc.inverse_transform(np.array([labelled_input[j][i]], dtype=np.int32))
                    temp[self.data_interface.feature_names[i]] = val[0]
                else:
                    temp[self.data_interface.feature_names[i]] = labelled_input[j][i]
            input_instance.append(temp)
        input_instance_df = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        return input_instance_df

    def label_decode_cfs(self, cfs_arr):
        ret_df = None
        if cfs_arr is None:
            return None
        for cf in cfs_arr:
            df = self.label_decode(cf)
            if ret_df is None:
                ret_df = df
            else:
                ret_df = pd.concat([ret_df, df])
        return ret_df

    def get_valid_feature_range(self, normalized=False):
        ret = self.data_interface.get_valid_feature_range(self.feature_range, normalized=normalized)
        for feat_name in self.data_interface.categorical_feature_names:
            # ret[feat_name] = self.labelencoder[feat_name].transform(ret[feat_name])
            ret[feat_name] = np.array(ret[feat_name], dtype=float)
        return ret

    def get_constraint_activities(self, d4py):
        activations = set()
        targets = set()
        for constraint in d4py.model.checkers:
            # str split here for binary constraints, otherwise append if supports cardinality
            if constraint['template'].supports_cardinality:
                activations.add(constraint['attributes'])
            elif constraint['template'].is_binary:
                str1, str2 = constraint['attributes'].replace(' ', '').split(',')
                activations.add(str1)
                targets.add(str2)
        targets = set([x for x in targets if x not in activations])
        activities = activations.copy()
        activities.update(targets)
        return activities, activations, targets

    def filter_declare_model(self, query_instance, encoder, d4py):
        query_instance_to_decode = pd.DataFrame(np.array(query_instance, dtype=float),
                                                columns=self.data_interface.feature_names)
        encoder.decode(query_instance_to_decode)

        query_instance_to_decode.insert(loc=0, column='Case ID',
                                        value=np.divmod(np.arange(len(query_instance_to_decode)), 1)[0] + 1)
        query_instance_to_decode.insert(loc=1, column='label', value=1)
        long_query_instance = pd.wide_to_long(query_instance_to_decode, stubnames=['prefix'], i='Case ID',
                                              j='order', sep='_', suffix=r'\w+')
        long_query_instance_sorted = long_query_instance.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        columns_to_rename = {'Case ID': 'case:concept:name', 'prefix': 'concept:name'}
        long_query_instance_sorted.rename(columns=columns_to_rename, inplace=True)
        long_query_instance_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
        long_query_instance_sorted.replace('0', 'other', inplace=True)
        query_log = convert_to_event_log(long_query_instance_sorted)
        d4py.load_xes_log(query_log)
        model_check_query = d4py.conformance_checking(consider_vacuity=False)
        query_patterns = {
            constraint
            for trace, patts in model_check_query.items()
            for constraint, checker in patts.items()
            if checker.state == TraceState.SATISFIED
        }

        def remove(list):
            pattern = r'(Exactly|Existence|Absence)(1|2|3)'
            # Replace "Exactly1" or "Existence1" with "Exactly" or "Existence"
            list = [re.sub(pattern, r'\1', s) for s in list]
            return list

        query_pattern = list(query_patterns)
        query_pattern_filter = remove(query_pattern)

        model_constraints = d4py.model.constraints
        updated_constraints = []
        indexes = []
        for i in model_constraints:
            if i in query_pattern_filter:
                indexes.append(d4py.model.constraints.index(i))
                updated_constraints.append(i)
        d4py.model.checkers = [d4py.model.checkers[i] for i in indexes]
        d4py.model.constraints = updated_constraints

    def compute_conformance_new(self, population, encoder, ltlf_model, dfa):
        population_df = pd.DataFrame(population, columns=self.data_interface.feature_names)
        encoder.decode(population_df)
        population_df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(population_df)), 1)[0] + 1)
        population_df.insert(loc=1, column='label', value=1)
        long_data = pd.wide_to_long(population_df, stubnames=['prefix'], i='Case ID',
                                    j='order', sep='_', suffix=r'\w+')
        timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
        long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
        long_data_sorted['time:timestamp'] = timestamps
        long_data_sorted['label'].replace({1: 'regular'}, inplace=True)
        long_data_sorted.drop(columns=['order'], inplace=True)
        columns_to_rename = {'Case ID': 'case:concept:name', 'prefix': 'concept:name'}
        long_data_sorted.rename(columns=columns_to_rename, inplace=True)
        long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
        long_data_sorted.replace('0', 'other', inplace=True)
        dataframe = format_dataframe(long_data_sorted, case_id='case:concept:name', activity_key='concept:name',
                                           timestamp_key='time:timestamp')
        log = convert_to_event_log(dataframe)
        event_log = D4PyEventLog()
        event_log.load_xes_log(log)
        analyzer = LTLAnalyzer(event_log, ltlf_model)
        start = time.time()
        jobs = 1
        conf_check_res_df = analyzer.run(
            jobs=jobs,
            dfa = dfa
        )
        end = time.time()
        print('Time taken for conformance checking with',jobs,'jobs', end - start)
        conf_check_res_df['accepted'].replace({True: 1, False: 0},inplace=True)
        conformance_score = pd.Series(data=conf_check_res_df['accepted'].values, index=conf_check_res_df['case:concept:name'].astype('int64')).sort_index()
        population_conformance = conf_check_res_df[conf_check_res_df['accepted'] == True].shape[0] / conf_check_res_df.shape[0]
        return conformance_score, population_conformance


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def check_state(trace,dfa,singleton,backend,original_activities,
                query_instance_activity):
    skip = True
    #print('Try')


    if backend == 'ltlf2dfa':
        current_states = {1}
    else:
        current_states = {dfa.initial_state}
    #current_states = {dfa.initial_state}
    for event in trace:
        temp = dict()
        symbol = str(event)
        symbol = Utils.parse_parenthesis(symbol)
        if skip:
            pass
        else:
            symbol = Utils.encode_attribute_type(attribute) + "_" + symbol
        symbol = Utils.parse_activity(symbol)
        if backend == 'lydia':
            symbol = symbol.lower()
        else:
            symbol = symbol.lower()
        temp[symbol] = True
        result_set = set()
        for x in current_states:
            successors = dfa.get_successors(x, temp)
            current_states = result_set.union(successors)
        #sink_state = all(is_sink(dfa, state) for state in current_states)
        #print('SymbolicDFA sink state', sink_state)

        #if sink_state:
        #    break
   # is_accepted = any(dfa.is_accepting(state) for state in current_states)
    final_states = singleton.accepting_states
    print('current_states',current_states)
    if current_states == {2}:
        print('STOP')
    temp = dict()
    temp[query_instance_activity] = True
    for x in current_states:
        successors = dfa.get_successors(x, temp)
    print('successors',successors)
    good_transitions = [item for item in singleton.get_transitions() if
                   item[0] == next(iter(current_states)) and item[2] in successors]
    good_transition_activities = [x[1] for x in good_transitions]
    include = [str(x).split("|&") for x in good_transition_activities if '~' not in str(x)]
    include = list(itertools.chain.from_iterable(include))
    exclude = [x for x in list(original_activities.values()) if x not in include]

    print('include',include)
    print('exclude',exclude)
    return include,exclude


from pythomata import SimpleDFA
from flloat.parser.ltlf import LTLfParser
import sys


def symbolic_to_singleton(symbolic_dfa, letters):
    states = symbolic_dfa.states
    initial_state = symbolic_dfa.initial_state
    accepting_states = symbolic_dfa.accepting_states

    transition_function = {}
    for s in states:
        transitions_of_s = {}
        for _, formula, s_1 in symbolic_dfa.get_transitions_from(s):
            for letter in letters:
                interpretation = {l: 0 for l in letters}
                interpretation[letter] = 1
                if formula.subs(interpretation):
                    transitions_of_s[letter] = s_1
        transition_function[s] = transitions_of_s

    singleton_dfa = SimpleDFA(states, letters, initial_state, accepting_states, transition_function)
    singleton_dfa = singleton_dfa  # .minimize().trim()
    singleton_dfa = singleton_dfa.complete()
    # graph = singleton_dfa.to_graphviz()
    # graph.render('/DFA')
    return singleton_dfa


def alphabet(parsed_formula):
    letters = parsed_formula.find_labels()
    return letters


def parse(formula):
    parser = LTLfParser()
    parsed_formula = parser(formula)
    return parsed_formula


def to_dfa(parsed_formula):
    symbolic_dfa = parsed_formula.to_automaton()
    return symbolic_dfa