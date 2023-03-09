import pandas as pd
import numpy as np
import bisect

class flip_data():

    def inverse_flip(poi_train_x, poi_train_y, poison_ct):
        #self.poi_train_x = poi_train_x
        #self.poi_train_y = poi_train_y
        #self.poison_ct = poison_ct

        # First calculates the dot product of the transpose of training set and training set using the np.dot() function
        dot_product = np.dot(poi_train_x.T, poi_train_x)
        # Then, it adds a scaled identity matrix to the resulting square matrix. The scaling factor is 0.01.
        # The identity matrix is created using the np.eye() function with a size equal to the number of columns in training set.
        scal_id_matrix = 0.01 * np.eye(poi_train_x.shape[1])
        # Calculates the inverse of a matrix inv_cov using the training set and the identity matrix np.eye().
        # Then takes the inverse of the resulting matrix using the ** -1 notation.
        # This results in the inv_cov matrix, which can be used in various linear algebraic operations.
        # The inverse of the covariance matrix is used in various algorithms to estimate the regression coefficients, compute prediction intervals, or perform principal component analysis, among other things.
        inv_cov = (dot_product + scal_id_matrix) ** -1
        # The resulting matrix is then multiplied by the transpose of poi_train_x using poi_train_x.T. This is equivalent to computing poi_train_x times inv_cov times poi_train_x transpose.
        # The resulting matrix H represents the projection of the training data onto a lower-dimensional space that captures the most important information or variance in the data
        H = poi_train_x @ inv_cov @ poi_train_x.T
        # row sum of H
        row_sum = np.sum(H, axis=1)
        train_y_arr = np.array(poi_train_y)
        # computes an auxiliary array that measures uncertainty in the target y
        y_uncert = np.abs(train_y_arr - 0.4) + 0.4
        '''
        num_differences = np.count_nonzero(train_y_arr != y_uncert)
        proportion_differences = num_differences / np.size(train_y_arr)
        print("difference ",proportion_differences)
        '''
        # computes an auxiliary array that represents the "flip" or opposite of the target y
        y_opp = 1 - np.floor(0.5 + train_y_arr)
        print("y_opp", y_opp)
        # combines the projection strength and uncertainty measures for each instance into a single statistic that captures the trade-off between projection quality and target uncertainty.
        # stat = np.multiply(bests.ravel(), room.ravel())
        stats = (y_uncert * y_opp).flatten()
        print("stats", stats)
        # Compute the total probability of all instances
        total_prob = sum(stats)
        print("total_prob", total_prob)
        # Initialize a list all_prob with a zero value.
        all_prob = [0]
        # Initialize an empty list to store the selected instance indices.
        poi_idx = []
        # for i in range(len(poi_train_x)):
        #    all_prob.append(stats[i] + all_prob[-1])
        all_prob = [sum(stats[:i + 1]) for i in range(poi_train_x.shape[0])]
        poi_idx = [bisect.bisect_left(all_prob, np.random.uniform(low=0, high=total_prob)) for i in range(poison_ct)]

        x_pois = poi_train_x[poi_idx]
        y_pois = [y_opp[i] for i in poi_idx]

        # print("x_pois: ", x_pois)
        print("x_pois len: ", len(x_pois))
        print("x_pois col ct:", x_pois.shape[1])
        # print("y_pois: ", y_pois)
        print("y_pois len: ", len(y_pois))

        return x_pois, y_pois
    
     def B_flip(poi_train_x, poi_train_y, poison_ct):
        poison_inds = np.random.choice(poi_train_x.shape[0], poison_ct, replace=False)
        x_poison = np.matrix(poi_train_x[poison_inds])
        y_poison = [1 if 1 - poi_train_y[i] > 0.5 else 0 for i in poison_inds]
        return x_poison, y_poison
