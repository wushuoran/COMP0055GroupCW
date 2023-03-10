import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

class poisoner(object):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):

        # this class deals with gradient descent and poisoning
        # validation set is used for validating and stop when goal reached
        # it takes in several hyperparameters which explained in ATTACK_* notebooks

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.valid_x = valid_x
        self.valid_y = valid_y

        self.sample_amt = train_x.shape[0]
        self.col_amt = train_x.shape[1]

        self.eta = eta
        self.beta = beta
        self.sigma = sigma
        self.eps = eps
        self.init_classifier, self.init_lam = None, None

    def poison_data(self, x_pois, y_pois, stop_1, stop_2, stop3, decrease_rate):

        '''
        Algorithm 1 in the paper
        Poisoning Attack Algorithm
        '''

        poison_ct = x_pois.shape[0]
        print("*****************************")
        print("**** Poison Count: ", poison_ct, " ****")
        print("*****************************")
        best_x_pois = np.zeros(x_pois.shape)
        best_y_pois = [None]
        best_obj = 0
        count = 0
        no_progress_count = 0

        sig = self.compute_sigma()
        multiplier = self.compute_multiplier()
        equation_7_left = np.bmat([[sig, np.transpose(multiplier)], [multiplier, np.matrix([1])]])
        # figure out starting error
        it_res = self.iteration_train(x_pois, y_pois, x_pois, y_pois)
        if it_res[0] > best_obj:
            best_x_pois = x_pois
            best_y_pois = y_pois
            best_obj = it_res[0]
        print("Iteration ", count, "\nObjective Value: ", it_res[0], " Change: ", it_res[0])
        print("Validation MSE ", it_res[2][0], "\nTest MSE ", it_res[2][1])
        print(" ")

        ''' Repeat (Line 3 in Algorithm 1)'''
        while True:
            count += 1
            new_x_pois = np.matrix(np.zeros(x_pois.shape))
            new_y_pois = [None for a in y_pois]
            x_cur = np.concatenate((self.train_x, x_pois), axis=0)
            y_cur = self.train_y + y_pois

            classifier, lam = self.learn_model(x_cur, y_cur, None)

            ''' for c = 1, ... , p do (Line 6 in Algorithm 1)'''
            for i in range(poison_ct): # deal with each poisoned data element
                # Poisons a single data point and generate the new point and  flag indicating
                x_pois_ele = x_pois[i]
                y_pois_ele = y_pois[i]
                m = self.compute_matrix(classifier, x_pois_ele, y_pois_ele)

                weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col \
                    = self.compute_weight_bias(equation_7_left, classifier.coef_, m, self.sample_amt, x_pois_ele)
                option_arg = (self.compute_vector_r(classifier, lam),)
                attack, attack_y = self.compute_attack_train \
                    (classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col, *option_arg)

                all_attacks = np.array(np.concatenate((attack, attack_y), axis=1))
                all_attacks = all_attacks.ravel()
                norm = np.linalg.norm(all_attacks)
                all_attacks = all_attacks / norm if norm > 0 else all_attacks
                ''' line search (Line 7 in Algorithm 1)'''
                x_pois_ele, y_pois_ele = self.search_line(x_pois_ele, y_pois_ele, all_attacks[:-1], all_attacks[-1])
                x_pois_ele = x_pois_ele.reshape((1, self.col_amt))
                new_x_pois[i] = x_pois_ele
                new_y_pois[i] = y_pois_ele

            # train the new model on the new poisoned data
            it_res = self.iteration_train(x_pois, y_pois, new_x_pois, new_y_pois)
            print("Iteration ", count)
            print("Objective Value:", it_res[0], " Difference: ", (it_res[0] - it_res[1]))

            if (it_res[0] <= it_res[1]):
                print("NO PROGRESS MADE!")
                no_progress_count += 1
                self.eta *= decrease_rate # reduce the learning rate
            else:
                no_progress_count = 0
                x_pois = new_x_pois
                y_pois = new_y_pois
            print(" ")
            # if the progress made is the best ever
            if (it_res[0] > best_obj):
                best_x_pois = x_pois
                best_y_pois = y_pois
                best_obj = it_res[1]

            ''' stopping conditions, until (Line 11 in Algorithm 1)'''
            it_diff = abs(it_res[0] - it_res[1])
            if count >= stop_1: # at least run 'stop1' iterations
                if (it_diff <= self.eps or count >= stop_2):
                    break   # if goal is reached or reach the maximum iteration limit 'stop2'
            if no_progress_count >= stop3: # stop if no progress is made after 'stop3' iterations
                break
        ''' OUTPUT final poisoning attack samples'''
        return best_x_pois, best_y_pois

    def search_line(self, x_pois_ele, y_pois_ele, attack, attack_y):
        k = 0
        x0 = np.copy(self.train_x)
        y0 = self.train_y[:]

        # Append the new point to the copy of the training data
        current_x = np.append(x0, x_pois_ele, axis=0)
        current_y = y0[:]
        current_y.append(y_pois_ele)

        # Train the model on the augmented data, then make a copy of the classifier
        classifier, lam = self.learn_model(current_x, current_y, None)
        classifier_copy = classifier
        # print("clf copy ",classifier_copy)
        # Initialize variables for tracking progress
        last_x_pois_ele = x_pois_ele
        current_x_pois_ele = x_pois_ele
        last_yc = y_pois_ele
        current_yc = y_pois_ele
        option_arg = None

        # Compute the objective function value before starting the line search
        obj_value_before = self.compute_obj_train(classifier, lam, option_arg)
        # Perform line search until convergence or max iterations reached
        count = 0
        eta = self.eta

        while True:
            if (count > 0):
                eta = self.beta * eta
            # Update the adversarial point and clip it to valid input range
            current_x_pois_ele = current_x_pois_ele + eta * attack
            current_x_pois_ele = np.clip(current_x_pois_ele, 0, 1)
            current_x[-1] = current_x_pois_ele
            # Update the adversarial label and clip it to valid label range
            current_yc = current_yc + attack_y * eta
            current_yc = min(1, max(0, current_yc))
            current_y[-1] = current_yc
            # Train the model on the updated data and compute the objective function value
            classifier_copy, lam1 = self.learn_model(current_x, current_y, classifier_copy)
            obj_value_after = self.compute_obj_train(classifier_copy, lam1, option_arg)

            # Check for convergence or bad progress
            if (count >= 99 or abs(obj_value_before - obj_value_after) < 1e-8):
                break
            if (obj_value_after - obj_value_before < 0):  # bad progress
                current_x_pois_ele = last_x_pois_ele
                current_yc = last_yc
                break
            # Update progress variables
            last_x_pois_ele = current_x_pois_ele
            last_yc = current_yc
            obj_value_before = obj_value_after
            k += 1
            count += 1

        return np.clip(current_x_pois_ele, 0, 1), current_yc

    def compute_error(self, classifier, plot, poisoned):
        # Compute predicted values
        test_y_pred = classifier.predict(self.test_x)
        valid_y_pred = classifier.predict(self.valid_x)
        if (plot is True):
            plt.scatter(self.test_y, test_y_pred)
            if (poisoned is True):
                plt.title("Flipped & Poisoned")
            else:
                plt.title("Flipped")
            plt.xlabel("Testing Data")
            plt.ylabel("Testing Predicted")
            plt.show()
            plt.scatter(self.valid_y, valid_y_pred)
            plt.xlabel("Validation Data")
            plt.ylabel("Validation Predicted")
            plt.show()
        # Compute squared errors
        test_mse = np.mean((test_y_pred - self.test_y) ** 2)
        valid_mse = np.mean((valid_y_pred - self.valid_y) ** 2)
        return valid_mse, test_mse

    def iteration_train(self, last_x_pois, last_y_pois, current_x_pois, current_y_pois):
        # Concatenate last x and y points with original data to create new training data
        x_train = np.concatenate((self.train_x, last_x_pois), axis=0)
        y_train = self.train_y + last_y_pois

        # Train a new model on the concatenated data
        classifier_last, lam_last = self.learn_model(x_train, y_train, None)

        # Compute the objective function value for the new model
        obj_last = self.compute_obj_train(classifier_last, lam_last, None)

        # Concatenate current x and y points with original data to create new training data
        x_train = np.concatenate((self.train_x, current_x_pois), axis=0)
        y_train = self.train_y + current_y_pois

        # Train a new model on the concatenated data
        classifier_current, lam_current = self.learn_model(x_train, y_train, None)

        # Compute the objective function value for the new model
        obj_current = self.compute_obj_train(classifier_current, lam_current, None)

        # Compute the error of the current model
        # Compute predicted values
        test_y_pred = classifier_current.predict(self.test_x)
        valid_y_pred = classifier_current.predict(self.valid_x)
        # Compute squared errors
        test_mse = np.mean((test_y_pred - self.test_y) ** 2)
        valid_mse = np.mean((valid_y_pred - self.valid_y) ** 2)
        error = tuple([valid_mse, test_mse])
        return obj_current, obj_last, error


class linear_poisoner(poisoner):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps)
        self.train_x = train_x
        self.train_y = train_y
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None)

    def learn_model(self, x, y, classifier):
        if not classifier:
            classifier = linear_model.LinearRegression()
        classifier.fit(x, y)
        return classifier, 0

    def compute_sigma(self):
        return np.dot(np.transpose(self.train_x), self.train_x) / self.train_x.shape[0]

    def compute_multiplier(self):
        return np.mean(self.train_x, axis=0)

    def compute_matrix(self, classifier, x_pois_ele, y_pois_ele):
        weight = classifier.coef_
        bias = classifier.intercept_
        x_pois_ele_transpose = np.reshape(x_pois_ele, (self.col_amt, 1))
        w_transpose = np.reshape(weight, (1, self.col_amt))
        err_ter_m = (np.dot(weight, x_pois_ele_transpose) + bias - y_pois_ele).reshape((1, 1))
        return np.dot(x_pois_ele_transpose, w_transpose) + err_ter_m[0, 0] * np.identity(self.col_amt)

    def compute_weight_bias(self, equation_7_left, w, m, n, x_pois_ele):
        equation_7_right = -(1 / n) * np.bmat([[m, -np.matrix(x_pois_ele.T)], [np.matrix(w.T), np.matrix([-1])]])

        weight_bias_matrix = np.linalg.lstsq(equation_7_left, equation_7_right, rcond=None)[0]
        weight_expt_last_row = weight_bias_matrix[:-1, :-1]  # get all but last row
        bias_last_row = weight_bias_matrix[-1, :-1]  # get last row
        weight_expt_last_col = weight_bias_matrix[:-1, -1]
        bias_last_col = weight_bias_matrix[-1, -1]

        return weight_expt_last_row, bias_last_row.ravel(), weight_expt_last_col.ravel(), bias_last_col

    def compute_vector_r(self, classifier, lam):
        # a zero vector of length equal to the number of features in the training data
        return np.zeros((1, self.col_amt))

    def compute_obj_train(self, classifier, lam, option_arg):
        errs = classifier.predict(self.train_x) - self.train_y
        return (np.linalg.norm(errs) ** 2) / self.sample_amt

    def compute_obj_valid(self, classifier, lam, option_arg):
        errs = classifier.predict(self.valid_x) - self.valid_y
        return (np.linalg.norm(errs) ** 2) / self.valid_x.shape[0]

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,option_arg):
        res = (classifier.predict(self.train_x) - self.train_y)

        grad_x = np.dot(self.train_x, weight_expt_last_row) + bias_last_row
        grad_y = np.dot(self.train_x, weight_expt_last_col.T) + bias_last_col

        attack_x = np.dot(res, grad_x) / self.sample_amt
        attack_y = np.dot(res, grad_y) / self.sample_amt

        if np.array_equal(option_arg, "normalized"):
            attack_norm = np.linalg.norm((attack_x, attack_y))
            if attack_norm > self.eps:
                attack_x = (attack_x / attack_norm) * self.eps
                attack_y = (attack_y / attack_norm) * self.eps

        return attack_x, attack_y

    def compute_attack_valid(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        n = self.valid_x.shape[0]
        res = (classifier.predict(self.valid_x) - self.valid_y)

        grad_x = np.dot(self.valid_x, weight_expt_last_row) + bias_last_row
        grad_y = np.dot(self.valid_x, weight_expt_last_col.T) + bias_last_col

        attack_x = np.dot(res, grad_x) / n
        attack_y = np.dot(res, grad_y) / n

        if np.array_equal(option_arg, "normalized"):
            attack_norm = np.linalg.norm((attack_x, attack_y))
            if attack_norm > self.eps:
                attack_x = (attack_x / attack_norm) * self.eps
                attack_y = (attack_y / attack_norm) * self.eps

        return attack_x, attack_y


class lasso_poisoner(linear_poisoner):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps)
        self.init_lam = -1
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None, None)

    def learn_model(self, x, y, classifier, lam=None):
        if (lam is None and self.init_lam != -1):
            lam = self.init_lam
        if (classifier is None):
            if (lam is None):
                classifier = linear_model.LassoCV(max_iter=10000)
                classifier.fit(x, y)
                lam = classifier.alpha_
            classifier = linear_model.Lasso(alpha=lam, max_iter=10000, warm_start=True)
        classifier.fit(x, y)
        return classifier, lam

    def compute_vector_r(self, classifier, lam):
        r = super().compute_vector_r(classifier, lam)
        r += lam * np.matrix(self.init_classifier.coef_).reshape(1, self.col_amt)
        return r

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col, option_arg):
        attack_x, attack_y = super().compute_attack_train(classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col,bias_last_col, option_arg)
        attack_x += self.init_lam * np.dot(option_arg, weight_expt_last_row)
        attack_y += self.init_lam * np.dot(option_arg, weight_expt_last_col.T)
        return attack_x, attack_y

    def compute_obj_train(self, classifier, lam, option_arg):
        curweight = super().compute_obj_train(classifier, lam, option_arg)
        l1_norm = np.linalg.norm(classifier.coef_, 1)
        #l1_norm += np.linalg.norm(self.init_classifier.coef_, 1)
        return lam * l1_norm + curweight


class ridge_poisoner(linear_poisoner):

    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps)
        # Initialize initial lambda and classifier
        self.init_lam = -1
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None, lam=None)

    def compute_obj_train(self, classifier, lam, option_arg):
        # Compute L2 norm of coefficients
        l2_norm = np.linalg.norm(classifier.coef_) / 2
        # Call parent class method to compute objective function value
        curweight = super().compute_obj_train(classifier, lam, option_arg)
        # Add L2 regularization term to the objective function
        return lam * l2_norm + curweight

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        r, = option_arg
        # Call parent class method to compute the poisoned data
        attack_x, attack_y = super().compute_attack_train(classifier, weight_expt_last_row, bias_last_row,
                                                     weight_expt_last_col, bias_last_col, option_arg)
        # Add regularization term to the attacked data
        attack_x += np.dot(r, weight_expt_last_row)
        attack_y += np.dot(r, weight_expt_last_col.T)
        return attack_x, attack_y

    def compute_vector_r(self, classifier, lam):
        # Call parent class method to compute r value
        r = super().compute_vector_r(classifier, lam)
        # Add L2 regularization term to r
        r += lam * np.matrix(classifier.coef_).reshape(1, self.col_amt)
        return r

    def compute_sigma(self):
        # Call parent class method to compute sigma
        basesigma = super().compute_sigma()
        # Add initial lambda to sigma
        sigma = basesigma + self.init_lam * np.eye(self.col_amt)
        return sigma

    def learn_model(self, x, y, classifier, lam=None):
        # Set lambda to 0.1 if not provided
        if lam is None:
            lam = 0.1
        # Initialize Ridge regression model
        classifier = linear_model.Ridge(alpha=lam, max_iter=10000)
        # Fit the model with given input data
        classifier.fit(x, y)
        # Return the trained model and lambda
        return classifier, lam



class e_net_poisoner(linear_poisoner):

    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):
        # Call parent class constructor
        poisoner.__init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps)
        # Initialize initial lambda and classifier
        self.init_lam = -1
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None, None)

    def compute_obj_train(self, classifier, lam, option_arg):
        l1_norm = np.linalg.norm(classifier.coef_, 1)
        l2_norm = np.linalg.norm(classifier.coef_, 2) / 2
        return (lam * (l1_norm + l2_norm)) / 2 + super().compute_obj_train(classifier, lam, option_arg)

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                             option_arg):
        attack_x, attack_y = super().compute_attack_train \
            (classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col, option_arg)
        attack_x += self.init_lam * np.dot(option_arg, weight_expt_last_row)
        attack_y += self.init_lam * np.dot(option_arg, weight_expt_last_col.T)
        return attack_x, attack_y

    def compute_vector_r(self, classifier, lam):
        errors = classifier.predict(self.train_x) - self.train_y
        l1 = (np.dot(errors, self.train_x) * (-1)) / self.sample_amt
        l2 = np.matrix(classifier.coef_).reshape(1, self.col_amt)
        return lam * (l1 + l2) * 0.5 + super().compute_vector_r(classifier, lam)

    def compute_sigma(self):
        # Add initial lambda to sigma
        return  self.init_lam * np.eye(self.col_amt) * 0.5 #+ super().compute_sigma()

    def learn_model(self, x, y, classifier, lam=None):
        if (lam is None and self.init_lam != -1):
            lam = self.init_lam
        if (classifier is None):
            if (lam is None):
                classifier = linear_model.ElasticNetCV(max_iter=10000)
                classifier.fit(x, y)
                lam = classifier.alpha_
            classifier = linear_model.ElasticNet(alpha=lam, max_iter=10000, warm_start=True)
            # classifier = linear_model.ElasticNetCV(max_iter=10000)
        classifier.fit(x, y)
        return classifier, lam