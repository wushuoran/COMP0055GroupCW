import numpy as np
import bisect

class flip_data():

    def inverse_flip(poi_train_x, poi_train_y, poison_ct):

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
        #print("y_opp", y_opp)

        stats = (y_uncert * y_opp).flatten()
        #print("stats", stats)
        # Compute the total probability of all instances
        total_prob = sum(stats)
        #print("total_prob", total_prob)
        all_prob = [sum(stats[:i + 1]) for i in range(poi_train_x.shape[0])]
        poi_idx = [bisect.bisect_left(all_prob, np.random.uniform(low=0, high=total_prob)) for i in range(poison_ct)]

        x_pois = poi_train_x[poi_idx]
        y_pois = [y_opp[i] for i in poi_idx]

        # print("x_pois: ", x_pois)
        #print("x_pois len: ", len(x_pois))
        #print("x_pois col ct:", x_pois.shape[1])
        # print("y_pois: ", y_pois)
        #print("y_pois len: ", len(y_pois))

        return x_pois, y_pois
    
    def B_flip(poi_train_x, poi_train_y, poison_ct):
        poison_idx = np.random.choice(poi_train_x.shape[0], poison_ct, replace=False)
        x_poison = np.matrix(poi_train_x[poison_idx])
        y_poison = [1 if 1 - poi_train_y[i] > 0.5 else 0 for i in poison_idx]
        return x_poison, y_poison
