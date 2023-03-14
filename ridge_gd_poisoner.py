import numpy as np
from sklearn import linear_model
from scipy.optimize import line_search


class poisoner(object):

    def __init__(self, train_x, train_y, test_x, test_y, valid_x, valid_y, step, beta, sigma, eps):

        # this class deals with gradient descent and poisoning
        # validation set is used for validating and stop when goal reached
        # it takes in several hyperparameters which explained in ATTACK_* notebooks

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.valid_x = valid_x
        self.valid_y = valid_y

        self.row_amt = train_x.shape[0]
        self.col_amt = train_x.shape[1]

        self.step = step
        self.beta = beta
        self.sigma = sigma
        self.eps = eps
        self.init_classifier = linear_model.Ridge(alpha=0.1, max_iter=10000)
        self.init_classifier.fit(self.train_x, self.train_y)

    def poison_data(self, x_pois, y_pois, stop_1, stop_2, stop3, decrease_rate):

        """
        Algorithm 1 in the paper
        Optimization-Based Poisoning Attack Algorithm (OptP)
        """

        poison_ct = x_pois.shape[0]
        # initialise two empty list in the shape of train_x and train_y to store best poison data
        best_x_pois = np.matrix(np.empty(x_pois.shape))
        best_y_pois = [None] * len(y_pois)
        best_loss = 0
        no_progress_count = 0

        ''' Iteration counter (Line 1 in Algorithm 1)'''
        count = 0

        ''' Get current model situation (Line 2 in Algorithm 1) '''
        loss_current, loss_previous, valid_mse, test_mse = self.iteration_train(x_pois, y_pois, x_pois, y_pois)
        if loss_current > best_loss: # compare with previous classifiers with lower poison rate
            best_x_pois = x_pois
            best_y_pois = y_pois
            best_loss = loss_current

        print("\n\n*****************************")
        print("**** Poison Count: ", poison_ct, " ****")
        print("*****************************")
        print("Initial Iteration", "\nCurrent Loss: ", loss_current)
        print("Validation MSE ", valid_mse, "\nTest MSE ", test_mse)
        print(" ")

        ''' Repeat (Line 3 in Algorithm 1)'''
        while True:
            # create two empty lists in the shape of train_x and train_y to store new poison data
            new_x_pois = np.matrix(np.empty(x_pois.shape))
            new_y_pois = [None] * len(y_pois)
            # Fit the model with given input data
            self.init_classifier.fit(np.vstack((self.train_x, x_pois)), self.train_y + y_pois)
            ''' for c = 1, ... , p do (Line 6 in Algorithm 1)'''
            for i in range(poison_ct):  # deal with each poisoned data element
                # Poisons a single data point and generate the new point and flag indicating
                x_pois_ele = x_pois[i]
                y_pois_ele = y_pois[i]
                ''' prepare parameters needed by line search '''
                weight = self.init_classifier.coef_
                bias = self.init_classifier.intercept_
                # converts the row vector into a column vector, used for matrix multiplication
                x_pois_ele_transpose = x_pois_ele.reshape(self.col_amt, 1)
                # converts the weight vector into a row vector, used for matrix multiplication
                w_transpose = weight.reshape(1, self.col_amt)
                # error between the predicted and actual output values for x_pois_ele
                error = (np.dot(weight, x_pois_ele_transpose) + bias - y_pois_ele).reshape((1, 1))
                ''' Appendix A, Theorem 3 in the paper '''
                hessian_matrix = np.matmul(x_pois_ele_transpose, w_transpose) + np.eye(self.col_amt) * error[0, 0]
                weight_expt_last_row, bias_last_row, weight_expt_last_col, bias_last_col \
                    = self.compute_weight_bias(weight, hessian_matrix, self.row_amt, x_pois_ele)
                # regularization parameter. used to prevent over-fitting
                reg = np.empty((1, self.col_amt)) # all set to 0, no regularization for OLS
                # Ridge uses L2 regularization term
                reg += 0.1 * np.matrix(weight).reshape(1, self.col_amt)
                result = (self.init_classifier.predict(self.train_x) - self.train_y)
                # calculate the gradients for the last rows and columns of the weight matrix
                grad_x = np.dot(self.train_x, weight_expt_last_row) + bias_last_row
                grad_y = np.dot(self.train_x, weight_expt_last_col.T) + bias_last_col
                # numpy arrays with shape (1, col_amt),
                # gradients of the loss function with respect to the weights for each row and column
                attack_x = np.dot(result, grad_x) / self.row_amt + np.dot(reg, weight_expt_last_row)
                attack_y = np.dot(result, grad_y) / self.row_amt + np.dot(reg, weight_expt_last_col.T)
                # concatenate above into a single array, then flattened into one-dimensional
                all_attacks = np.array(np.hstack((attack_x, attack_y))).flatten()
                # normalize the attacks by dividing them by Euclidean norm
                # only perform the division when norm > 0
                norm = np.linalg.norm(all_attacks)
                if norm > 0:
                    all_attacks = all_attacks / norm
                ''' line search (Line 7 in Algorithm 1)'''
                x_pois_ele, y_pois_ele = self.line_search(x_pois_ele, y_pois_ele, all_attacks[:-1], all_attacks[-1])
                x_pois_ele = x_pois_ele.reshape((1, self.col_amt))
                # assign the poisoned row and label to the lists created at the beginning of while loop
                new_x_pois[i] = x_pois_ele
                new_y_pois[i] = y_pois_ele

            # train the new model on the new poisoned data
            loss_current, loss_previous, valid_mse, test_mse = self.iteration_train(x_pois, y_pois, new_x_pois, new_y_pois)
            print("Iteration ", count + 1)
            print("Loss:", loss_current, " Difference: ", (loss_current - loss_previous))

            if (loss_current <= loss_previous):
                print("NO PROGRESS MADE!")
                no_progress_count += 1
                self.step *= decrease_rate  # reduce the learning rate
            else:
                no_progress_count = 0
                x_pois = new_x_pois
                y_pois = new_y_pois
            print(" ")
            # if the progress made is the best ever
            if (loss_current > best_loss):
                best_x_pois = x_pois
                best_y_pois = y_pois
                best_loss = loss_current

            ''' count increment (Line 10 in Algorithm 1)'''
            count += 1
            ''' stopping conditions, until (Line 11 in Algorithm 1)'''
            diff = abs(loss_current - loss_previous)
            if count >= stop_1:  # at least run 'stop1' iterations
                if (diff <= self.eps or count >= stop_2):
                    break  # if goal is reached or reach the maximum iteration limit 'stop2'
            if no_progress_count >= stop3:  # stop if no progress is made after 'stop3' iterations
                break

        ''' OUTPUT final poisoning attack samples'''
        return best_x_pois, best_y_pois

    def line_search(self, x_pois_ele, y_pois_ele, attack_vals, attack_bias):
        """ optimise the content of current poisoned row and its label, to maximise the impact
            Reference: 'lineSearch' in author's code """
        count = 0
        step = self.step
        train_x_copy = self.train_x
        train_y_copy = self.train_y
        # Append the new point to the copy of the training data
        current_x = np.append(train_x_copy, x_pois_ele, axis=0)
        current_y = np.append(train_y_copy, y_pois_ele)
        # Train the model on the poisoned data, then make a copy of the classifier
        classifier_copy = self.init_classifier
        # Initialize variables for tracking progress
        last_x_pois_ele = x_pois_ele
        last_y_pois_ele = y_pois_ele
        current_x_pois_ele = x_pois_ele
        current_y_pois_ele = y_pois_ele
        # Compute the loss value before starting the line search
        loss_before = self.loss_function(classifier_copy)
        # Perform line search until convergence or max iterations reached
        while True:
            if count > 0:
                step = self.beta * step
            # Update the adversarial point and clip it to valid input range
            current_x_pois_ele = current_x_pois_ele + step * attack_vals
            current_x_pois_ele = np.minimum(current_x_pois_ele,1)
            current_x_pois_ele = np.maximum(current_x_pois_ele,0)
            current_x[-1] = current_x_pois_ele
            # Update the adversarial label and clip it to valid label range
            current_y_pois_ele = current_y_pois_ele + attack_bias * step
            current_y_pois_ele = np.minimum(current_y_pois_ele, 1)
            current_y_pois_ele = np.maximum(current_y_pois_ele, 0)
            current_y[-1] = current_y_pois_ele
            # Train the model on the updated data and compute the objective function value
            classifier_copy.fit(current_x, current_y)
            loss_after = self.loss_function(classifier_copy)
            # Check for convergence or bad progress
            if count >= 99 or abs(loss_before - loss_after) < 1e-8:
                break
            if loss_after - loss_before < 0:  # bad progress
                current_x_pois_ele = last_x_pois_ele
                current_y_pois_ele = last_y_pois_ele
                break
            # Update progress variables
            loss_before = loss_after
            last_x_pois_ele = current_x_pois_ele
            last_y_pois_ele = current_y_pois_ele
            count += 1
        return current_x_pois_ele, current_y_pois_ele

    def loss_function(self, classifier):
        """ a regularized loss function computed on the training data (excluding the poisoning points) """
        # Compute L2 norm of coefficients
        l2 = np.linalg.norm(classifier.coef_) / 2
        # compute objective function value
        errs = classifier.predict(self.train_x) - self.train_y
        current_weight = (np.linalg.norm(errs) ** 2) / self.row_amt
        # Add L2 regularization term to the objective function
        return 0.1 * l2 + current_weight

    def iteration_train(self, last_x_pois, last_y_pois, current_x_pois, current_y_pois):
        """ compare the effectiveness of the current poisoned data with the previous one """
        # Concatenate last x and y points with original data to create new training data
        # Train a new model on the concatenated data
        classifier_last = self.init_classifier
        classifier_last.fit(np.vstack((self.train_x, last_x_pois)), self.train_y + last_y_pois)
        # Compute the objective function value for the new model
        loss_previous = self.loss_function(classifier_last)
        # Concatenate current x and y points with original data to create new training data
        # Train a new model on the concatenated data
        classifier_current = self.init_classifier
        classifier_current.fit(np.vstack((self.train_x, current_x_pois)), self.train_y + current_y_pois)
        # Compute the objective function value for the new model
        loss_current = self.loss_function(classifier_current)
        # Compute the error of the current model
        # Compute predicted values
        test_y_pred = classifier_current.predict(self.test_x)
        valid_y_pred = classifier_current.predict(self.valid_x)
        # Compute squared errors
        test_mse = np.mean((test_y_pred - self.test_y) ** 2)
        valid_mse = np.mean((valid_y_pred - self.valid_y) ** 2)
        return loss_current, loss_previous, valid_mse, test_mse

    def compute_weight_bias(self, weight, matrix, row_amt, x_pois_ele):
        """ uses equation 4 & 7 in the paper.
            Reference: 'compute_wb_zc' in authors' code """
        multiplier = np.mean(self.train_x, axis=0)
        # Add initial lambda to sigma
        init_sigma = np.dot(np.transpose(self.train_x), self.train_x) / self.train_x.shape[0]
        final_sigma = init_sigma + 0.1 * np.eye(self.col_amt)
        ' first element of equation 4 right hand side '
        equation_7_left = np.bmat([[final_sigma, np.transpose(multiplier)], [multiplier, np.matrix([1])]])
        equation_7_right = -(1 / row_amt) * np.bmat([[matrix, -np.matrix(x_pois_ele.T)], [np.matrix(weight.T), np.matrix([-1])]])

        weight_bias_matrix = np.linalg.lstsq(equation_7_left, equation_7_right, rcond=None)[0]
        weight_expt_last_row = weight_bias_matrix[:-1, :-1]  # get all but last row
        bias_last_row = weight_bias_matrix[-1, :-1]  # get last row
        weight_expt_last_col = weight_bias_matrix[:-1, -1]
        bias_last_col = weight_bias_matrix[-1, -1]

        return weight_expt_last_row, bias_last_row.ravel(), weight_expt_last_col.ravel(), bias_last_col
