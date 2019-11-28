import numpy as np
import random
import matplotlib.pyplot as plt

class Applicant_group:
    def __init__(self, name, color, line_style, size, scores, loan_demand, error_rate, score_error, market, repays, interest_rate_limit=np.inf):
        self.name = name
        self.color = color
        self.line_style = line_style
        self.size = size
        self.scores = scores
        self.real_scores = self.set_real_scores(error_rate, score_error, market)
        self.sort_scores()
        self.initial_mean_score = np.mean(scores)
        self.loan_demand = loan_demand
        self.score_repay_prob = self.get_repay_prob_mapping(market.score_range, repays)
        self.interest_rate_limit = interest_rate_limit
        
    def get_repay_prob_mapping(self, score_range, repay_prob):
        x_axis = np.linspace(score_range[0],score_range[1],score_range[1]-score_range[0]+1, dtype=int)
        y_axis = np.interp(x_axis, repay_prob.index, repay_prob[self.name])
        return dict(zip(x_axis, y_axis))
    
        #get simulated repay outcome of score probability(1=repaid, 0=default )
    def get_repay_outcome(self, repay_probability):
        random_number = random.random()
        outcome = 1
        #print(random_number)
        if random_number < 1-repay_probability:
            outcome = 0
        return outcome
    
        #simulating that some members of the group have better score/repay prob then rated
    def set_real_scores(self, error_rate, score_error, market):
        real_scores = self.scores.copy()
        better_applicants = np.sort(random.sample(range(0, self.size), int(self.size*error_rate)))
        for applicant in better_applicants:
            if real_scores[applicant] + score_error < market.score_range[0]:
                real_scores[applicant] = market.score_range[0]
            elif real_scores[applicant] + score_error > market.score_range[1]:
                real_scores[applicant] = market.score_range[1]
            else:
                real_scores[applicant] += score_error
        return real_scores
    
    def get_mean_score_change(self):
            return np.mean(self.scores)-self.initial_mean_score
    
    def sort_scores(self):
        #np.set_printoptions(threshold=sys.maxsize)
        sorted_indices = np.argsort(self.scores[::-1])
        sorted_scores = np.zeros(len(sorted_indices), dtype = int) 
        sorted_real_scores = np.zeros(len(sorted_indices), dtype = int) 
        for i in range(0, len(sorted_indices)): 
            sorted_scores[::-1][i] = self.scores[::-1][sorted_indices[i]]
            sorted_real_scores[::-1][i] = self.real_scores[::-1][sorted_indices[i]]
        self.scores = sorted_scores
        self.real_scores = sorted_real_scores
        return self.scores
    
    def select_score_change(self, market, outcome):
        if outcome:
            return market.repay_score
        else:
            return market.default_score
    
    def change_score(self, market, applicant_index, outcome):
        score_change = self.select_score_change(market, outcome)
        if self.scores[applicant_index] + score_change < market.score_range[0]:
            self.scores[applicant_index] = market.score_range[0]
        elif self.scores[applicant_index] + score_change > market.score_range[1]:
            self.scores[applicant_index] = market.score_range[1]
        else:
            self.scores[applicant_index] += score_change
        
        return self.scores[applicant_index]
    
    def change_real_score(self, market, applicant_index, outcome):
        score_change = self.select_score_change(market, outcome)
        if self.real_scores[applicant_index] + score_change < market.score_range[0]:
            self.real_scores[applicant_index] = market.score_range[0]
        elif self.real_scores[applicant_index] + score_change > market.score_range[1]:
            self.real_scores[applicant_index] = market.score_range[1]
        else:
            self.real_scores[applicant_index] += score_change
        
        return self.real_scores[applicant_index]
    
    def plot_histogram(self, market):
        plt.figure()
        plt.hist(self.scores, range = market.score_range, label='Step ' + str(market.step))
        plt.title(self.name + " group histogram change for " + market.policy + " policy")
        plt.ylabel("Occurence")
        plt.xlabel("Score")
        plt.ylim([0,self.size*0.75])
        plt.legend(loc="upper left")
        plt.show()
        return 1
        
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
