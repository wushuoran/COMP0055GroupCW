import numpy as np
from sklearn import linear_model


class poisoner(object):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, \
                 eta, beta, sigma, eps):

        """
        GDPoisoner handles gradient descent and poisoning routines
        Computations for specific models found in respective classes

        x, y: training set
        testx, testy: test set
        validx, validy: validation set used for evaluation and stopping]
        """

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.valid_x = valid_x
        self.valid_y = valid_y

        self.sample_amt = train_x.shape[0]
        self.col_amt = train_x.shape[1]

        self.attack_comp = self.compute_attack_train
        self.obj_comp = self.compute_obj_train

        self.eta = eta
        self.beta = beta
        self.sigma = sigma
        self.eps = eps
        self.init_classifier, self.init_lam = None, None

    def poison_data(self, x_pois, y_pois):

        """
        poison_data takes an initial set of poisoning points and optimizes it
        using gradient descent with parameters set in __init__
        """
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
        it_res = self.iteration_progress(x_pois, y_pois, x_pois, y_pois)

        print("Iteration {}:".format(count))
        print("Objective Value: {} Change: {}".format(it_res[0], it_res[0]))
        print("Validation MSE: {}".format(it_res[2][0]))
        print("Test MSE: {}".format(it_res[2][1]))
        print(" ")
        if it_res[0] > best_obj:
            best_x_pois, best_y_pois, best_obj = x_pois, y_pois, it_res[0]

        # main work loop
        while True:
            count += 1
            new_x_pois = np.matrix(np.zeros(x_pois.shape))
            new_y_pois = [None for a in y_pois]
            x_cur = np.concatenate((self.train_x, x_pois), axis=0)
            y_cur = self.train_y + y_pois

            classifier, lam = self.learn_model(x_cur, y_cur, None)

            for i in range(poison_ct):
                # Poisons a single data point and generate the new point and  flag indicating
                x_pois_ele = x_pois[i]
                y_pois_ele = y_pois[i]
                m = self.compute_matrix(classifier,x_pois_ele,y_pois_ele)

                weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col = self.compute_weight_bias(
                    equation_7_left, classifier.coef_, m, self.sample_amt, x_pois_ele)
                option_arg = (self.compute_vector_r(classifier, lam),) #if self.objective == 0 else ()
                attack, attack_y = self.attack_comp(classifier, weight_expt_last_row, bias_last_row,\
                                                    weight_expt_last_col,\
                                                    bias_last_col, *option_arg)

                all_attacks = np.array(np.concatenate((attack, attack_y), axis=1))
                all_attacks = all_attacks.ravel()
                norm = np.linalg.norm(all_attacks)
                all_attacks = all_attacks / norm if norm > 0 else all_attacks

                x_pois_ele, y_pois_ele, _ = self.search_line(x_pois_ele, y_pois_ele, all_attacks[:-1], all_attacks[-1])
                x_pois_ele = x_pois_ele.reshape((1, self.col_amt))
                new_x_pois[i] = x_pois_ele
                new_y_pois[i] = y_pois_ele
            it_res = self.iteration_progress(x_pois, y_pois, new_x_pois, new_y_pois)
            print("Iteration ", count)
            print("Objective Value:", it_res[0], " Difference: ", (it_res[0] - it_res[1]))

            # if no progress is made, reduce the learning rate
            if (it_res[0] < it_res[1]):
                print("NO PROGRESS MADE!")
                self.eta *= 0.75
            else:
                x_pois = new_x_pois
                y_pois = new_y_pois
            print(" ")
            if (it_res[0] > best_obj):
                best_x_pois, best_y_pois, best_obj = x_pois, y_pois, it_res[1]

            it_diff = abs(it_res[0] - it_res[1])

            # stopping conditions
            if count >= 20:
                if (it_diff <= self.eps or count >= 30):
                    break

        return best_x_pois, best_y_pois



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

        current_x = np.delete(current_x, current_x.shape[0] - 1, axis=0)
        current_x = np.append(current_x, current_x_pois_ele, axis=0)
        current_y[-1] = current_yc
        classifier1, lam1 = self.learn_model(current_x, current_y, None)

        obj_value_after = self.obj_comp(classifier1, lam1, option_arg)

        return np.clip(current_x_pois_ele, 0, 1), current_yc, obj_value_after

    def compute_error(self, classifier):
        # Compute predicted values
        test_y_pred = classifier.predict(self.test_x)
        valid_y_pred = classifier.predict(self.valid_x)
        # Compute squared errors
        test_mse = np.mean((test_y_pred - self.test_y) ** 2)
        valid_mse = np.mean((valid_y_pred - self.valid_y) ** 2)

        return valid_mse, test_mse
    def iteration_progress(self, last_x_pois, last_y_pois, current_x_pois, current_y_pois):
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
        # Compute predicted values
        test_y_pred = classifier_current.predict(self.test_x)
        valid_y_pred = classifier_current.predict(self.valid_x)
        # Compute squared errors
        test_mse = np.mean((test_y_pred - self.test_y) ** 2)
        valid_mse = np.mean((valid_y_pred - self.valid_y) ** 2)
        error = tuple([valid_mse , test_mse])
        return obj_current, obj_last, error

class linear_poisoner(poisoner):
    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):
        # Call parent class constructor
        super().__init__(train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps)
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

    def compute_obj_train(self, classifier, lam, option_arg):
        errs = classifier.predict(self.train_x) - self.train_y
        mse = np.linalg.norm(errs) ** 2 / self.sample_amt
        return mse

    def compute_obj_valid(self, classifier, lam, option_arg):
        m = self.valid_x.shape[0]
        errs = classifier.predict(self.valid_x) - self.valid_y
        mse = np.linalg.norm(errs) ** 2 / m
        return mse

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
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

    def compute_vector_r(self, classifier, lam):
        r = super().compute_vector_r(classifier, lam)
        if self.init_classifier is not None:
            r += lam * np.matrix(self.init_classifier.coef_).reshape(1, self.col_amt)
        return r

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        r, = option_arg
        attack_x, attack_y = super().compute_attack_train(classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col,
                                                     bias_last_col, option_arg)
        if self.init_classifier is not None:
            attack_x += self.init_lam * np.dot(r, weight_expt_last_row)
            attack_y += self.init_lam * np.dot(r, weight_expt_last_col.T)
        return attack_x, attack_y

    def compute_obj_train(self, classifier, lam, option_arg):
        curweight = super().compute_obj_train(classifier, lam, option_arg)
        l1_norm = np.linalg.norm(classifier.coef_, 1)
        if self.init_classifier is not None:
            l1_norm += np.linalg.norm(self.init_classifier.coef_, 1)
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
'''
class e_net_poisoner(poisoner):

    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps):
        # Call parent class constructor
        poisoner.__init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, eta, beta, sigma, eps)
        # Initialize initial lambda and classifier
        self.init_lam = -1
        self.init_classifier, self.init_lam = self.learn_model(self.train_x, self.train_y, None, None)

    def compute_obj_train(self, clf, lam, otherargs):
        curweight = linear_model.comp_W_0(self, clf, lam, otherargs)

        l1_norm = np.linalg.norm(clf.coef_, 1)
        l2_norm = np.linalg.norm(clf.coef_, 2) / 2
        sum = l1_norm + l2_norm

        return (lam * sum) / 2 #+ curweight

    def compute_attack_train(self, classifier, weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col,
                        option_arg):
        r, = option_arg
        # Call parent class method to compute the poisoned data
        attack_x, attack_y = linear_poisoner.compute_attack_train(classifier, weight_expt_last_row, bias_last_row,
                                                     weight_expt_last_col, bias_last_col, option_arg)
        # Add regularization term to the attacked data
        attack_x += np.dot(r, weight_expt_last_row)
        attack_y += np.dot(r, weight_expt_last_col.T)
        return attack_x, attack_y

    def compute_vector_r(self, classifier, lam):
        # Call parent class method to compute r value
        r = np.zeros((1, self.col_amt))

        errors = classifier.predict(self.train_x) - self.train_y
        l1 = (np.dot(errors, self.train_x)*(-1)) / self.sample_amt
        l2 = np.matrix(classifier.coef_).reshape(1, self.col_amt)

        r = lam * (l1 + l2) * 0.5
        return r

    def compute_sigma(self):
        # Add initial lambda to sigma
        sigma = self.init_lam * np.eye(self.col_amt) * 0.5
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

    def learn_model(self, x, y, classifier, lam=None):
        print("x into classfier:",x)
        if (lam is None):
            lam = self.init_lam
            print("lam val: ", lam)
        if (classifier is None):
            print("classifier is None")
            if (lam is not None):
                print("lam now has value, enter sub if")
                classifier = linear_model.ElasticNetCV(max_iter=10000)
                print("classifier now assigned")
                classifier.fit(x,y)
                print("fit data")
                lam = classifier.alpha_
                print("new lam = ", lam)
            classifier = linear_model.ElasticNet(alpha=lam, max_iter=10000,warm_start = True)
            print("outer classifier reached")
        print("second x into classfier:", x)
        classifier.fit(x,y)
        print("final fit")
        return classifier, lam
'''