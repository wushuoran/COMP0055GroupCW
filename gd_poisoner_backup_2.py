import numpy as np
import datetime
from sklearn import linear_model


class poisoner(object):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, \
                 eta, beta, sigma, eps, train_file, result_file, column_map):

        """
        GDPoisoner handles gradient descent and poisoning routines
        Computations for specific models found in respective classes

        x, y: training set
        testx, testy: test set
        validx, validy: validation set used for evaluation and stopping

        train_file: file storing poisoning points in each iteration
        result_file: file storing each iteration's results in csv format
        """

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.valid_x = valid_x
        self.valid_y = valid_y

        self.sample_amt = train_x.shape[0]
        self.col_amt = train_x.shape[1]

        self.objective = 0

        self.attack_comp = self.comp_attack_trn
        self.obj_comp = self.comp_obj_trn

        self.eta = eta
        self.beta = beta
        self.sigma = sigma
        self.eps = eps

        self.train_file = train_file
        self.result_file = result_file
        self.init_classifier, self.init_lam = None, None

        self.column_map = column_map

    def poison_data(self, x_pois, y_pois):

        """
        poison_data takes an initial set of poisoning points and optimizes it
        using gradient descent with parameters set in __init__
        """
        tstart = datetime.datetime.now()
        poison_ct = x_pois.shape[0]
        print("***************************")
        print("**** Poison Count: {} ****".format(poison_ct))
        print("***************************")
        best_x_pois = np.zeros(x_pois.shape)
        best_y_pois = [None for a in y_pois]

        best_obj = 0

        count = 0

        sig = self.compute_sigma()  # can already compute sigma and multiplier
        multiplier = self.compute_multiplier()
        equation_7_left = np.bmat([[sig, np.transpose(multiplier)], [multiplier, np.matrix([1])]])

        # figure out starting error
        it_res = self.iter_progress(x_pois, y_pois, x_pois, y_pois)

        print("Iteration {}:".format(count))
        print("Objective Value: {} Change: {}".format(it_res[0], it_res[0]))
        print("Validation MSE: {}".format(it_res[2][0]))
        print("Test MSE: {}".format(it_res[2][1]))
        print(" ")
        if it_res[0] > best_obj:
            best_x_pois, best_y_pois, best_obj = x_pois, y_pois, it_res[0]

        self.train_file.write("\n")
        self.train_file.write(str(poison_ct) + "," + str(count) + '\n')

        for j in range(poison_ct):
            self.train_file.write(','.join([str(val) for val in [y_pois[j]] + x_pois[j].tolist()[0]]) + '\n')

        # main work loop
        while True:
            count += 1
            new_x_pois = np.matrix(np.zeros(x_pois.shape))
            new_y_pois = [None for a in y_pois]
            x_cur = np.concatenate((self.train_x, x_pois), axis=0)
            y_cur = self.train_y + y_pois

            classifier, lam = self.learn_model(x_cur, y_cur, None)
            pois_params = [(x_pois[i], y_pois[i], equation_7_left, classifier, lam) for i in range(poison_ct)]
            out_bound_ct = 0

            for i in range(poison_ct):
                cur_pois_res = self.poison_data_subroutine(pois_params[i])
                new_x_pois[i] = cur_pois_res[0]
                new_y_pois[i] = cur_pois_res[1]
                out_bound_ct += cur_pois_res[2]

            it_res = self.iter_progress(x_pois, y_pois, new_x_pois, new_y_pois)

            print("Iteration ", count)
            print("Objective Value:", it_res[0], " Difference: ", (it_res[0] - it_res[1]))

            # if we don't make progress, decrease learning rate
            if (it_res[0] < it_res[1]):
                print("NO PROGRESS MADE!")
                self.eta *= 0.75
                new_x_pois, new_y_pois = x_pois, y_pois
            else:
                x_pois = new_x_pois
                y_pois = new_y_pois
            print(" ")
            if (it_res[0] > best_obj):
                best_x_pois, best_y_pois, best_obj = x_pois, y_pois, it_res[1]

            towrite = [poison_ct, count, it_res[0], it_res[1] - it_res[0], it_res[2][0], it_res[2][1],
                       (datetime.datetime.now() - tstart).total_seconds()]
            self.result_file.write(','.join([str(val) for val in towrite]) + "\n")
            self.train_file.write("\n{},{}\n".format(poison_ct, count))

            for j in range(poison_ct):
                self.train_file.write(','.join([str(val) for val in
                                                [new_y_pois[j]] + new_x_pois[j].tolist()[0]
                                                ]) + '\n')
            it_diff = abs(it_res[0] - it_res[1])

            # stopping conditions
            if count >= 20:
                if (it_diff <= self.eps or count >= 30):
                    break

        return best_x_pois, best_y_pois

    def poison_data_subroutine(self, in_tuple):
        """
        Poisons a single data point and returns the new point and a flag indicating
        whether the new point is out of bounds.

        Parameters:
            in_tuple: tuple of (x_pois_ele, y_pois_ele, equation_7_left, mu, classifier, lam)
                - x_pois_ele: numpy array of shape (1, self.col_amt) representing
                  the feature values of the point to be poisoned
                - y_pois_ele: float representing the true label of the point to be
                  poisoned
                - equation_7_left: numpy array of shape (self.col_amt,) representing the left
                  hand side of Equation 7 from the paper
                - mu: float representing the value of the Lagrange multiplier mu
                - classifier: sklearn linear model representing the current model
                - lam: float representing the regularization coefficient lambda

        Returns:
            Tuple of (numpy array of shape (1, self.col_amt), float, bool) representing
            the new feature values, new label, and a flag indicating whether the new
            point is out of bounds.
        """
        x_pois_ele, y_pois_ele, equation_7_left, classifier, lam = in_tuple
        m = self.compute_matrix(classifier, x_pois_ele, y_pois_ele)

        # compute partials
        weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col = self.compute_weight_bias(
            equation_7_left, classifier.coef_, m, self.sample_amt, x_pois_ele)

        option_arg = (self.compute_vector_r(classifier, lam),) if self.objective == 0 else ()

        attack, attack_y = self.attack_comp(classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col,
                                            bias_last_col, *option_arg)

        # keep track of how many points are pushed out of bounds
        if (y_pois_ele >= 1 and attack_y >= 0) or (y_pois_ele <= 0 and attack_y <= 0):
            outofbounds = True
        else:
            outofbounds = False

        all_attacks = np.array(np.concatenate((attack, attack_y), axis=1))
        all_attacks = all_attacks.ravel()
        norm = np.linalg.norm(all_attacks)
        all_attacks = all_attacks / norm if norm > 0 else all_attacks

        x_pois_ele, y_pois_ele, _ = self.search_line(x_pois_ele, y_pois_ele, all_attacks[:-1], all_attacks[-1])
        x_pois_ele = x_pois_ele.reshape((1, self.col_amt))

        return x_pois_ele, y_pois_ele, outofbounds

    def compute_error(self, classifier):
        # Compute predicted values
        test_y_pred = classifier.predict(self.test_x)
        valid_y_pred = classifier.predict(self.valid_x)
        # Compute squared errors
        test_mse = np.mean((test_y_pred - self.test_y) ** 2)
        valid_mse = np.mean((valid_y_pred - self.valid_y) ** 2)

        return valid_mse, test_mse

    def search_line(self, x_pois_ele, y_pois_ele, attack, attack_y):
        k = 0
        x0 = np.copy(self.train_x)
        y0 = self.train_y[:]

        # Append the new point to the copy of the training data
        current_x = np.append(x0, x_pois_ele, axis=0)
        current_y = y0[:]
        current_y.append(y_pois_ele)

        # Train the model on the augmented data
        classifier, lam = self.learn_model(current_x, current_y, None)
        classifier1, lam1 = classifier, lam

        # Initialize variables for tracking progress
        last_x_pois_ele = x_pois_ele
        current_x_pois_ele = x_pois_ele
        last_yc = y_pois_ele
        current_yc = y_pois_ele
        option_arg = None

        # Compute the objective function value before starting the line search
        obj_value_before = self.obj_comp(classifier, lam, option_arg)
        # Perform line search until convergence or max iterations reached
        count = 0
        eta = self.eta

        while True:
            if (count > 0):
                eta = self.beta * eta
            count += 1
            # Update the adversarial point and clip it to valid input range
            current_x_pois_ele = current_x_pois_ele + eta * attack
            current_x_pois_ele = np.clip(current_x_pois_ele, 0, 1)
            current_x[-1] = current_x_pois_ele
            # Update the adversarial label and clip it to valid label range
            current_yc = current_yc + attack_y * eta
            current_yc = min(1, max(0, current_yc))
            current_y[-1] = current_yc
            # Train the model on the updated data and compute the objective function value
            classifier1, lam1 = self.learn_model(current_x, current_y, classifier1)
            obj_value_after = self.obj_comp(classifier1, lam1, option_arg)

            # Check for convergence or bad progress
            if (count >= 100 or abs(obj_value_before - obj_value_after) < 1e-8):
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

        # Find the most important features and set them to 1, the rest to 0
        if self.column_map:
            for col in self.column_map:
                vals = [(current_x_pois_ele[0, j], j) for j in self.column_map[col]]
                topval, topcol = max(vals)
                for j in self.column_map[col]:
                    if (j != topcol):
                        current_x_pois_ele[0, j] = 0
                if (topval > 1 / (1 + len(self.column_map[col]))):
                    current_x_pois_ele[0, topcol] = 1
                else:
                    current_x_pois_ele[0, topcol] = 0
        current_x = np.delete(current_x, current_x.shape[0] - 1, axis=0)
        current_x = np.append(current_x, current_x_pois_ele, axis=0)
        current_y[-1] = current_yc
        classifier1, lam1 = self.learn_model(current_x, current_y, None)

        obj_value_after = self.obj_comp(classifier1, lam1, option_arg)

        return np.clip(current_x_pois_ele, 0, 1), current_yc, obj_value_after

    def iter_progress(self, last_x_pois, last_y_pois, current_x_pois, current_y_pois):
        # Concatenate last x and y points with original data to create new training data
        x_train = np.concatenate((self.train_x, last_x_pois), axis=0)
        y_train = self.train_y + last_y_pois

        # Train a new model on the concatenated data
        classifier_last, lam_last = self.learn_model(x_train, y_train, None)

        # Compute the objective function value for the new model
        obj_last = self.obj_comp(classifier_last, lam_last, None)

        # Concatenate current x and y points with original data to create new training data
        x_train = np.concatenate((self.train_x, current_x_pois), axis=0)
        y_train = self.train_y + current_y_pois

        # Train a new model on the concatenated data
        classifier_current, lam_current = self.learn_model(x_train, y_train, None)

        # Compute the objective function value for the new model
        obj_current = self.obj_comp(classifier_current, lam_current, None)

        # Compute the error of the current model
        error = self.compute_error(classifier_current)

        return obj_current, obj_last, error


class linear_poisoner(poisoner):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps, train_file,
                 result_file, column_map):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps, train_file,
                         result_file, column_map)
        self.train_x = train_x
        self.train_y = train_y
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None)

    def learn_model(self, x, y, classifier):
        if (not classifier):
            classifier = linear_model.LinearRegression()
        classifier.fit(x, y)
        return classifier, 0

    def compute_sigma(self):
        sigma = np.dot(np.transpose(self.train_x), self.train_x)
        sigma = sigma / self.train_x.shape[0]
        return sigma

    def compute_multiplier(self):
        multiplier = np.mean(self.train_x, axis=0)
        return multiplier

    def compute_matrix(self, classifier, x_pois_ele, y_pois_ele):
        w, b = classifier.coef_, classifier.intercept_
        x_pois_eletransp = np.reshape(x_pois_ele, (self.col_amt, 1))
        wtransp = np.reshape(w, (1, self.col_amt))
        errterm = (np.dot(w, x_pois_eletransp) + b - y_pois_ele).reshape((1, 1))
        first = np.dot(x_pois_eletransp, wtransp)
        m = first + errterm[0, 0] * np.identity(self.col_amt)
        return m

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

    def comp_obj_trn(self, classifier, lam, option_arg):
        errs = classifier.predict(self.train_x) - self.train_y
        mse = np.linalg.norm(errs) ** 2 / self.sample_amt
        return mse

    def comp_obj_vld(self, classifier, lam, option_arg):
        m = self.valid_x.shape[0]
        errs = classifier.predict(self.valid_x) - self.valid_y
        mse = np.linalg.norm(errs) ** 2 / m
        return mse

    def comp_attack_trn(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        res = (classifier.predict(self.train_x) - self.train_y)

        gradx = np.dot(self.train_x, weight_expt_last_row) + bias_last_row
        grady = np.dot(self.train_x, weight_expt_last_col.T) + bias_last_col

        attack_x = np.dot(res, gradx) / self.sample_amt
        attack_y = np.dot(res, grady) / self.sample_amt

        if np.array_equal(option_arg, "normalized"):
            attack_norm = np.linalg.norm((attack_x, attack_y))
            if attack_norm > self.eps:
                attack_x = (attack_x / attack_norm) * self.eps
                attack_y = (attack_y / attack_norm) * self.eps

        return attack_x, attack_y

    def comp_attack_vld(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        n = self.valid_x.shape[0]
        res = (classifier.predict(self.valid_x) - self.valid_y)

        gradx = np.dot(self.valid_x, weight_expt_last_row) + bias_last_row
        grady = np.dot(self.valid_x, weight_expt_last_col.T) + bias_last_col

        attack_x = np.dot(res, gradx) / n
        attack_y = np.dot(res, grady) / n

        if np.array_equal(option_arg, "normalized"):
            attack_norm = np.linalg.norm((attack_x, attack_y))
            if attack_norm > self.eps:
                attack_x = (attack_x / attack_norm) * self.eps
                attack_y = (attack_y / attack_norm) * self.eps

        return attack_x, attack_y


class lasso_poisoner(linear_poisoner):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps, train_file,
                 result_file, column_map):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps, train_file,
                         result_file, column_map)
        self.init_lam = None
        self.init_classifier = None

    def learn_model(self, x, y, classifier, lam=None):
        if lam is None and self.init_lam is not None:
            lam = self.init_lam
        if classifier is None:
            if lam is None:
                classifier = linear_model.LassoCV(max_iter=10000)
                classifier.fit(x, y)
                lam = classifier.alpha_
            classifier = linear_model.Lasso(alpha=lam, max_iter=10000, warm_start=True)
        classifier.fit(x, y)
        if self.init_classifier is None:
            self.init_classifier = classifier
            self.init_lam = lam
        return classifier, lam

    def compute_vector_r(self, clf, lam):
        r = super().compute_vector_r(clf, lam)
        if self.init_classifier is not None:
            r += lam * np.matrix(self.init_classifier.coef_).reshape(1, self.col_amt)
        return r

    def comp_attack_trn(self, clf, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        r, = option_arg
        attack_x, attack_y = super().comp_attack_trn(clf, weight_expt_last_row, bias_last_row, weight_expt_last_col,
                                                     bias_last_col, option_arg)
        if self.init_classifier is not None:
            attack_x += self.init_lam * np.dot(r, weight_expt_last_row)
            attack_y += self.init_lam * np.dot(r, weight_expt_last_col.T)
        return attack_x, attack_y

    def comp_obj_trn(self, clf, lam, option_arg):
        curweight = super().comp_obj_trn(clf, lam, option_arg)
        l1_norm = np.linalg.norm(clf.coef_, 1)
        if self.init_classifier is not None:
            l1_norm += np.linalg.norm(self.init_classifier.coef_, 1)
        return lam * l1_norm + curweight


class ridge_poisoner(linear_poisoner):

    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps, train_file,
                 result_file, column_map):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps, train_file,
                         result_file, column_map)
        # Initialize initial lambda and classifier
        self.init_lam = -1
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None, lam=None)

    def comp_obj_trn(self, classifier, lam, option_arg):
        # Compute L2 norm of coefficients
        l2_norm = np.linalg.norm(classifier.coef_) / 2
        # Call parent class method to compute objective function value
        curweight = super().comp_obj_trn(classifier, lam, option_arg)
        # Add L2 regularization term to the objective function
        return lam * l2_norm + curweight

    def comp_attack_trn(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        r, = option_arg
        # Call parent class method to compute the poisoned data
        attack_x, attack_y = super().comp_attack_trn(classifier, weight_expt_last_row, bias_last_row,
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
